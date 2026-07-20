# Lab 110 - In-World Interference (do counted chances combine like arrows?)
#
# Backward-arc, engineering-free: no Lab-99/101c theta anywhere. A two-path
# state z(phi) = cA*e^{i2piw}*uA + cB*e^{i2piw}*e^{i*phi}*uB is prepared and
# read out by argmax-counting only; Q_k(phi) (the quadratic ensemble average)
# is a target-side reference, never touching the counting path. Tier 1 =
# ensemble MC scout; Tier 2 = the same preparation realized inside a
# deterministic Model-Two-style world (certified pool, full-displacement fold).

import sys
import json
import time
from pathlib import Path

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB_FOLDER = Path(__file__).parent
LAB101_DIR = Path(__file__).resolve().parents[1] / "101_existence_hardening"
sys.path.insert(0, str(LAB101_DIR))
import existence_hardening_101 as lab101  # noqa: E402

lab91 = lab101.lab91
lab93 = lab101.lab93
lab100v2 = lab101.lab100v2
K = lab101.K

TWO_PI = 2 * np.pi
CBAND_LO, CBAND_HI = 0.55, 1.05
N_TIER1 = 200000
N_REF = 500000
REF_SEED = 11001
A2_SEEDS = (12001, 12002)
A2_NOISE_BAR = 0.005
T_TIER2 = 60000
T_TRANSIENT = 1000
PHI_GRID = np.linspace(0.0, TWO_PI, 24, endpoint=False)
TIER2_PHIS = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]

D_FLAT_BAR = 0.10
RMS_BAR = 0.03
OVERTONE_BAR = 0.10
TIER2_MATCH_BAR = 0.012
FIT_RESIDUAL_BAR = 0.002


def build_pairs(Qf):
    phi0, phi1, phi2 = Qf[:, 0], Qf[:, 1], Qf[:, 2]
    P1 = {"uA": phi0, "uB": phi1}
    P2 = {"uA": phi0, "uB": (phi1 + phi2) / np.sqrt(2)}
    return {"P1": P1, "P2": P2}


def draw_batch(uA, uB, U_S, Qf, phi, N, seed=None, rng=None, cA=None, cB=None, w=None):
    if rng is not None and cA is None:
        cA = rng.uniform(CBAND_LO, CBAND_HI, size=N)
        cB = rng.uniform(CBAND_LO, CBAND_HI, size=N)
        w = rng.uniform(0.0, 1.0, size=N)
    elif seed is not None:
        rng2 = np.random.default_rng(seed)
        cA = rng2.uniform(CBAND_LO, CBAND_HI, size=N)
        cB = rng2.uniform(CBAND_LO, CBAND_HI, size=N)
        w = rng2.uniform(0.0, 1.0, size=N)
    common = np.exp(2j * np.pi * w)
    z = (cA * common)[:, None] * uA[None, :] + (cB * common * np.exp(1j * phi))[:, None] * uB[None, :]
    y = z @ U_S.T
    overlaps = np.abs(y @ np.conj(Qf)) ** 2
    znorm2 = np.sum(np.abs(z) ** 2, axis=1)
    return overlaps, znorm2, cA, cB, w


def counted_P(uA, uB, U_S, Qf, phi, N, seed):
    overlaps, znorm2, *_ = draw_batch(uA, uB, U_S, Qf, phi, N, seed=seed)
    outcomes = np.argmax(overlaps, axis=1)
    counts = np.bincount(outcomes, minlength=3).astype(float)
    return counts / counts.sum()


def classical_null(uA, uB, U_S, Qf):
    # single-path ensembles: c and w have zero effect on argmax (global scale
    # and global phase). P_A, P_B are one-hot; the classical mix weights by
    # E[c^2], which is identical for A and B here (same iid c-distribution) -> 0.5/0.5.
    for u in (uA, uB):
        pass
    yA = (U_S @ uA)
    yB = (U_S @ uB)
    kA = int(np.argmax(np.abs(np.conj(Qf).T @ yA) ** 2))
    kB = int(np.argmax(np.abs(np.conj(Qf).T @ yB) ** 2))
    P_A = np.zeros(3)
    P_A[kA] = 1.0
    P_B = np.zeros(3)
    P_B[kB] = 1.0
    P_cls = 0.5 * P_A + 0.5 * P_B
    return P_cls, kA, kB


def fit_first_harmonic(phi_grid, F_k):
    X = np.column_stack([np.ones_like(phi_grid), np.cos(phi_grid), np.sin(phi_grid)])
    coeffs, *_ = np.linalg.lstsq(X, F_k, rcond=None)
    a0, ac, as_ = coeffs
    beta = np.hypot(ac, as_)
    gamma = np.arctan2(-as_, ac)
    fitted = a0 + ac * np.cos(phi_grid) + as_ * np.sin(phi_grid)
    resid = float(np.sqrt(np.mean((F_k - fitted) ** 2)))
    visibility = beta / a0 if a0 != 0 else np.inf
    return {"alpha": float(a0), "beta": float(beta), "gamma": float(gamma), "resid": resid,
            "visibility": float(visibility), "fitted": fitted.tolist()}


def overtone_ratio(F_k, noise_floor):
    centered = F_k - F_k.mean()
    fft_vals = np.fft.rfft(centered)
    power = np.abs(fft_vals) ** 2
    h1_power = float(power[1]) if len(power) > 1 else 0.0
    overtone_power = float(power[2:].sum()) if len(power) > 2 else 0.0
    underpowered = bool(h1_power < 10 * (noise_floor ** 2))
    R = overtone_power / h1_power if h1_power > 0 else float("inf")
    return {"h1_power": h1_power, "overtone_power": overtone_power, "R": R, "underpowered": underpowered}


if __name__ == "__main__":
    t_start = time.time()
    print("=" * 60)
    print("STAGE P - Proof-ette")
    print("=" * 60)
    print("Claim: |a + e^{i phi} b|^2 = |a|^2 + |b|^2 + 2 Re(e^{i phi} conj(a) b)")
    a_s, b_s, phi_s = sp.symbols('a b phi', complex=True)
    phi_r = sp.symbols('phi_r', real=True)
    expr = sp.Abs(a_s + sp.exp(sp.I * phi_r) * b_s) ** 2
    expanded = sp.expand(sp.re(expr.rewrite(sp.cos)))
    # direct symbolic check via conjugate expansion (works for symbolic complex a,b)
    lhs = (a_s + sp.exp(sp.I * phi_r) * b_s) * sp.conjugate(a_s + sp.exp(sp.I * phi_r) * b_s)
    rhs = a_s * sp.conjugate(a_s) + b_s * sp.conjugate(b_s) + \
        sp.exp(sp.I * phi_r) * sp.conjugate(a_s) * b_s + sp.exp(-sp.I * phi_r) * a_s * sp.conjugate(b_s)
    diff = sp.simplify(sp.expand(lhs - rhs))
    symbolic_ok = bool(diff == 0)
    print(f"  |a+e^i(phi)b|^2 - [|a|^2+|b|^2+2Re(e^i(phi) conj(a) b)] simplifies to 0: {symbolic_ok}")
    print("  => phi-dependence is EXACTLY the single term 2Re(e^{i phi} conj(a) b) = const*cos(phi+gamma):")
    print("     no phi^2, no cos(2phi), no higher harmonic possible - a pure first-harmonic fringe.")
    print("  Linearity of expectation over the (c_A,c_B,w) ensemble preserves this exactly (gamma_k is")
    print("  amplitude-independent - see report - so E[...] cannot introduce new phi-dependence.)")

    FREQS3, pool_integers = lab100v2.build_pool_v2()
    THETAS0 = lab100v2.build_thetas0(K)
    with open(LAB101_DIR.parents[0] / "100_repreparation_fold" / "metrics_100_v2.json") as f:
        m100v2 = json.load(f)
    stored_integers = m100v2["stageA"]["bridge"]["pool_integers_used"]

    Qf91 = lab91.haar_frame(91)
    pairs91 = build_pairs(Qf91)
    fit_residuals = {}
    for pair_name, pair in pairs91.items():
        Q_all = np.empty((24, 3))
        rng = np.random.default_rng(REF_SEED)
        cA = rng.uniform(CBAND_LO, CBAND_HI, size=N_REF)
        cB = rng.uniform(CBAND_LO, CBAND_HI, size=N_REF)
        w = rng.uniform(0.0, 1.0, size=N_REF)
        for i, phi in enumerate(PHI_GRID):
            overlaps, znorm2, *_ = draw_batch(pair["uA"], pair["uB"], lab93.U_S, Qf91, phi, N_REF,
                                              cA=cA, cB=cB, w=w)
            probs = overlaps / znorm2[:, None]
            Q_all[i] = probs.mean(axis=0)
        for k in range(3):
            fit = fit_first_harmonic(PHI_GRID, Q_all[:, k])
            fit_residuals[f"91|{pair_name}|k{k}"] = fit["resid"]
    max_fit_resid = max(fit_residuals.values())
    p1_symbolic_ok = symbolic_ok
    p1_fit_ok = bool(max_fit_resid < FIT_RESIDUAL_BAR)
    print(f"\n  Numerical check: max 1st-harmonic fit residual over Q_k grids = {max_fit_resid:.6f} "
          f"(<{FIT_RESIDUAL_BAR}), pass={p1_fit_ok}")

    stageP = {"symbolic_ok": p1_symbolic_ok, "fit_residuals": fit_residuals,
              "max_fit_residual": max_fit_resid, "fit_ok": p1_fit_ok}

    print("\n" + "=" * 60)
    print("STAGE A - Gates")
    print("=" * 60)
    integers_match = bool(pool_integers == stored_integers)
    M_born91 = lab101.compute_M_born_gen(Qf91, lab93.U_S)
    mborn_dev = float(np.max(np.abs(lab93.compute_M_born(Qf91) - M_born91)))
    A1 = bool(integers_match and mborn_dev < 1e-12)
    print(f"A1 (machinery): integers_match={integers_match}, mborn_dev={mborn_dev:.3e} (<1e-12), pass={A1}")

    P1 = pairs91["P1"]
    P_a = counted_P(P1["uA"], P1["uB"], lab93.U_S, Qf91, 0.0, N_TIER1, A2_SEEDS[0])
    P_b = counted_P(P1["uA"], P1["uB"], lab93.U_S, Qf91, 0.0, N_TIER1, A2_SEEDS[1])
    a2_max_diff = float(np.max(np.abs(P_a - P_b)))
    A2 = bool(a2_max_diff < A2_NOISE_BAR)
    print(f"A2 (instrument sanity, phi=0/P1, two 200k batches): max|diff|={a2_max_diff:.5f} "
          f"(<{A2_NOISE_BAR}), pass={A2}")
    noise_floor = a2_max_diff

    pass_a = bool(A1 and A2)
    stageA = {"integers_match": integers_match, "mborn_dev": mborn_dev,
              "a2_max_diff": a2_max_diff, "noise_floor": noise_floor, "pass": pass_a}
    print(f"\nStage A: {'PASS' if pass_a else 'FAIL'}")

    if not pass_a:
        metrics = {
            "lab": 110, "stream": "cryptographic-substrate", "kind": "scout+proof",
            "serves_node": "born_rule_emergence",
            "stageP": stageP, "stageA": stageA, "stageB": {}, "stageC": {}, "stageD": {},
            "verdict": "VOID", "verdict_reason": f"Stage A failed: A1={A1}, A2={A2}.",
            "leakage_audit": {
                "wrote_outside_lab_folder": False, "quadratic_form_on_empirical_side": False,
                "born_theta_used": False, "posthoc_threshold_change": False, "metrics_hand_edited": False,
            },
        }
        with open(LAB_FOLDER / "metrics_110.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nVerdict: VOID")
        sys.exit(0)

    # -----------------------------------------------------------------
    # Stage B - Tier 1: the fringe map
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE B - Tier 1: the fringe map")
    print("=" * 60)
    Qf30001 = lab91.haar_frame(30001)
    pairs30001 = build_pairs(Qf30001)
    cells = [(91, "P1", Qf91, pairs91["P1"]), (91, "P2", Qf91, pairs91["P2"]),
             (30001, "P1", Qf30001, pairs30001["P1"])]

    stageB = {}
    idx = 0
    for frame, pair_name, Qf, pair in cells:
        key = f"{frame}|{pair_name}"
        print(f"-- {key} --")
        F_all = np.empty((24, 3))
        for i, phi in enumerate(PHI_GRID):
            F_all[i] = counted_P(pair["uA"], pair["uB"], lab93.U_S, Qf, phi, N_TIER1, 12000 + idx)
            idx += 1

        Q_all = np.empty((24, 3))
        rng = np.random.default_rng(REF_SEED)
        cA = rng.uniform(CBAND_LO, CBAND_HI, size=N_REF)
        cB = rng.uniform(CBAND_LO, CBAND_HI, size=N_REF)
        w = rng.uniform(0.0, 1.0, size=N_REF)
        for i, phi in enumerate(PHI_GRID):
            overlaps, znorm2, *_ = draw_batch(pair["uA"], pair["uB"], lab93.U_S, Qf, phi, N_REF,
                                              cA=cA, cB=cB, w=w)
            Q_all[i] = (overlaps / znorm2[:, None]).mean(axis=0)

        P_cls, kA, kB = classical_null(pair["uA"], pair["uB"], lab93.U_S, Qf)

        F_bar = F_all.mean(axis=0)
        D_flat = float(np.max([0.5 * np.sum(np.abs(F_all[i] - F_bar)) for i in range(24)]))

        rms_per_k = [float(np.sqrt(np.mean((F_all[:, k] - Q_all[:, k]) ** 2))) for k in range(3)]
        rms_overall = max(rms_per_k)

        overtone_per_k = [overtone_ratio(F_all[:, k], noise_floor) for k in range(3)]
        max_R = max(o["R"] for o in overtone_per_k if not o["underpowered"]) if \
            any(not o["underpowered"] for o in overtone_per_k) else None
        any_underpowered = any(o["underpowered"] for o in overtone_per_k)

        fit_counted = [fit_first_harmonic(PHI_GRID, F_all[:, k]) for k in range(3)]
        fit_quad = [fit_first_harmonic(PHI_GRID, Q_all[:, k]) for k in range(3)]

        fringes_exist = bool(D_flat > D_FLAT_BAR)
        shape_match = bool(rms_overall <= RMS_BAR)
        purity_ok = bool(max_R is not None and max_R <= OVERTONE_BAR)

        print(f"   D_flat={D_flat:.4f} (fringes_exist={fringes_exist}), RMS={rms_overall:.4f} "
              f"(shape_match={shape_match}), max_R={max_R}, underpowered={any_underpowered} "
              f"(purity_ok={purity_ok})")
        for k in range(3):
            print(f"     k={k}: V_counted={fit_counted[k]['visibility']:.4f}, "
                  f"V_quad={fit_quad[k]['visibility']:.4f}, R_k={overtone_per_k[k]['R']:.4f}, "
                  f"underpowered={overtone_per_k[k]['underpowered']}")

        stageB[key] = {
            "frame": frame, "pair": pair_name, "F": F_all.tolist(), "Q": Q_all.tolist(),
            "P_classical": P_cls.tolist(), "classical_winners": {"A": kA, "B": kB},
            "D_flat": D_flat, "fringes_exist": fringes_exist,
            "rms_per_k": rms_per_k, "rms_overall": rms_overall, "shape_match": shape_match,
            "overtone_per_k": overtone_per_k, "max_R": max_R, "any_underpowered": any_underpowered,
            "purity_ok": purity_ok,
            "fit_counted": fit_counted, "fit_quad": fit_quad,
        }

    # -----------------------------------------------------------------
    # Stage C - Tier 2: the embedded (in-world) fringes
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE C - Tier 2: the embedded world")
    print("=" * 60)

    def run_tier2(uA, uB, Qf, U_S, z0, K_, FREQS, THETAS0, T, T_transient, phi):
        z = z0.copy()
        thetas = THETAS0.copy()
        outcomes = np.empty(T, dtype=np.int64)
        for t in range(T):
            znorm = np.linalg.norm(z)
            overlaps = np.array([np.abs(np.vdot(Qf[:, k], z)) ** 2 for k in range(3)]) / (znorm ** 2)
            j = int(np.argmax(overlaps))
            outcomes[t] = j
            gen_idx = t % K_
            phases6 = thetas[gen_idx]
            v_cA, v_cB, v_w = phases6[0] / TWO_PI, phases6[1] / TWO_PI, phases6[2] / TWO_PI
            cA = CBAND_LO + (CBAND_HI - CBAND_LO) * v_cA
            cB = CBAND_LO + (CBAND_HI - CBAND_LO) * v_cB
            common = np.exp(2j * np.pi * v_w)
            z_new = cA * common * uA + cB * common * np.exp(1j * phi) * uB
            z = U_S @ z_new
            thetas = np.mod(thetas + FREQS, TWO_PI)
        return outcomes

    stageC = {}
    tier2_rows = []
    P1_91 = pairs91["P1"]
    for phi in TIER2_PHIS:
        outcomes = run_tier2(P1_91["uA"], P1_91["uB"], Qf91, lab93.U_S, lab93.z0, K, FREQS3, THETAS0,
                              T_TIER2, T_TRANSIENT, phi)
        kept = outcomes[T_TRANSIENT:]
        counts = np.bincount(kept, minlength=3).astype(float)
        P_embedded = counts / counts.sum()
        phi_idx = int(round(phi / (TWO_PI / 24)))
        F_tier1_at_phi = np.array(stageB["91|P1"]["F"][phi_idx])
        max_diff = float(np.max(np.abs(P_embedded - F_tier1_at_phi)))
        row_match = bool(max_diff <= TIER2_MATCH_BAR)
        tier2_rows.append({"phi": float(phi), "phi_idx": phi_idx, "P_embedded": P_embedded.tolist(),
                           "F_tier1": F_tier1_at_phi.tolist(), "max_diff": max_diff, "match": row_match})
        print(f"  phi={phi:.4f} (grid idx {phi_idx}): P_embedded={np.round(P_embedded,4).tolist()}, "
              f"F_tier1={np.round(F_tier1_at_phi,4).tolist()}, max_diff={max_diff:.4f}, match={row_match}")

    tier2_all_match = bool(all(r["match"] for r in tier2_rows))
    print(f"Tier 2 overall match: {tier2_all_match}")
    stageC = {"rows": tier2_rows, "all_match": tier2_all_match}

    # -----------------------------------------------------------------
    # Stage D - P1-P3 scoring and verdict
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE D - Verdict inputs")
    print("=" * 60)
    p1_holds = bool(all(stageB[k]["fringes_exist"] for k in stageB))
    p2_holds = bool(all(stageB[k]["shape_match"] and stageB[k]["purity_ok"] for k in stageB))
    p3_holds = tier2_all_match
    print(f"P1 (fringes exist, all cells): {p1_holds}")
    print(f"P2 (shape match AND purity, all cells): {p2_holds}")
    print(f"P3 (Tier 2 matches Tier 1): {p3_holds}")

    all_fringes_exist = all(stageB[k]["fringes_exist"] for k in stageB)
    any_no_fringes = any(not stageB[k]["fringes_exist"] for k in stageB)
    all_shape_purity_ok = all(stageB[k]["shape_match"] and stageB[k]["purity_ok"] for k in stageB)

    if any_no_fringes:
        verdict = "NO-FRINGES"
        offenders = [k for k in stageB if not stageB[k]["fringes_exist"]]
        verdict_reason = f"D_flat <= {D_FLAT_BAR} for: {offenders}. Counted records ignore relative phase."
    elif all_fringes_exist and all_shape_purity_ok and tier2_all_match:
        verdict = "SIGNATURE-MATCH"
        verdict_reason = ("Fringes exist, shape and harmonic purity match the quadratic reference on all "
                           "three (frame,pair) cells, and Tier 2's embedded world matches Tier 1.")
    elif all_fringes_exist:
        verdict = "FRINGES-WITH-DEPARTURE"
        fails = [k for k in stageB if not (stageB[k]["shape_match"] and stageB[k]["purity_ok"])]
        if not tier2_all_match:
            fails = fails + ["Tier2-mismatch"]
        verdict_reason = f"Fringes exist everywhere, but shape/purity/Tier2 fails at: {fails}."
    else:
        verdict = "OTHER"
        verdict_reason = "No ladder rule matched exactly."

    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")

    # -----------------------------------------------------------------
    # The money plot
    # -----------------------------------------------------------------
    print("\nGenerating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, key in zip(axes, ["91|P1", "91|P2", "30001|P1"]):
        cell = stageB[key]
        F = np.array(cell["F"])
        Q = np.array(cell["Q"])
        P_cls = np.array(cell["P_classical"])
        for k in range(3):
            ax.scatter(PHI_GRID, F[:, k], s=18, label=f"counted k={k}")
            phi_fine = np.linspace(0, TWO_PI, 200)
            fit = cell["fit_quad"][k]
            q_fine = fit["alpha"] + fit["beta"] * np.cos(phi_fine + fit["gamma"])
            ax.plot(phi_fine, q_fine, "--", alpha=0.6)
            ax.axhline(P_cls[k], linestyle=":", alpha=0.3, color="gray")
        ax.set_xlabel("phi")
        ax.set_ylabel("P(k | phi)")
        ax.set_title(f"{key}: counted (dots) vs quadratic ref (dashed), classical (dotted)")
        ax.legend(fontsize=6)
    fig.suptitle("Lab 110: the two-slit inside the record world")
    fig.tight_layout()
    fig.savefig(LAB_FOLDER / "fringes_110.png", dpi=110)
    plt.close(fig)

    def strip_for_json(d):
        return json.loads(json.dumps(d, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))

    metrics = {
        "lab": 110, "stream": "cryptographic-substrate", "kind": "scout+proof",
        "serves_node": "born_rule_emergence",
        "stageP": strip_for_json(stageP),
        "stageA": strip_for_json(stageA),
        "stageB": strip_for_json(stageB),
        "stageC": strip_for_json(stageC),
        "stageD": {"p1_holds": p1_holds, "p2_holds": p2_holds, "p3_holds": p3_holds},
        "verdict": verdict, "verdict_reason": verdict_reason,
        "leakage_audit": {
            "wrote_outside_lab_folder": False, "quadratic_form_on_empirical_side": False,
            "born_theta_used": False, "posthoc_threshold_change": False, "metrics_hand_edited": False,
        },
    }
    with open(LAB_FOLDER / "metrics_110.json", "w") as f:
        json.dump(metrics, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")
