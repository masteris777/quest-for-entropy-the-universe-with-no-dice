# Lab 101b - Control Recalibration Addendum (C-passive transportability)
#
# Does NOT rerun, reinterpret, or amend any Lab 101 measurement. Lab 101's OTHER
# verdict stands frozen. This addendum asks a new, predeclared question about
# the two flagged C-passive cells.

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

METRICS_101_PATH = LAB_FOLDER / "metrics_101.json"

T_NULL = 20000
T_TRANSIENT = 1000
W1_SEEDS = list(range(20000, 20300))
W2_SEEDS = list(range(21000, 21300))
RATIO_BAR = 5.0
A2_TOL = 0.05
ANALYST_PRECOMPUTED_RATIOS = [95.0, 19.9, 28.7, 28.1, 48.8, 13.5]
RATIO_REPRODUCE_TOL = 1.0  # analyst's values are rounded to 1 decimal

FLAGGED_CELLS = {"W1": (1, 42), "W2": (2, 23)}


# ---------------------------------------------------------------------------
# The passive trajectory is frame-independent (z_{t+1} = U_S z_t from the
# world's z0, with no Qf dependence at all) - compute it ONCE per world, then
# classify+count per frame (vectorized, cheap).
# ---------------------------------------------------------------------------
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


def run_null_battery(U_S, z0, seeds, T, T_transient):
    z_hist = compute_passive_trajectory(U_S, z0, T)
    znorms_sq = np.sum(np.abs(z_hist) ** 2, axis=1)
    D_borns = []
    for seed in seeds:
        Qf = lab91.haar_frame(seed)
        M_born = lab101.compute_M_born_gen(Qf, U_S)
        D_borns.append(classify_D_born(z_hist, znorms_sq, Qf, M_born, T_transient))
    return np.array(D_borns), z_hist, znorms_sq


# ---------------------------------------------------------------------------
# Stage A - Gates
# ---------------------------------------------------------------------------
def run_stage_a():
    with open(METRICS_101_PATH) as f:
        m101 = json.load(f)
    controls = m101["stageE"]["controls"]
    all_cells_J = {tuple(c["cell"]): c["J"] for c in m101["stageB"] + m101["stageC"]}

    ratios = []
    for cell_tuple in lab101.CELLS:
        cell_list = list(cell_tuple)
        passive_val = next(c["value"] for c in controls if c["cell"] == cell_list and c["control"] == "C-passive")
        J_main = all_cells_J[cell_tuple]
        ratio = passive_val / J_main
        ratios.append({"cell": cell_list, "passive_D_born": passive_val, "J_main": J_main, "ratio": ratio})
        print(f"R1 {cell_tuple}: passive_D_born={passive_val:.5f}, J_main={J_main:.5f}, ratio={ratio:.2f}")

    my_ratios = [r["ratio"] for r in ratios]
    max_disagreement = float(max(abs(a - b) for a, b in zip(my_ratios, ANALYST_PRECOMPUTED_RATIOS)))
    A1 = bool(max_disagreement < RATIO_REPRODUCE_TOL)
    R1_holds = bool(all(r["ratio"] >= RATIO_BAR for r in ratios))
    print(f"A1: max_disagreement_with_analyst={max_disagreement:.2f}, pass={A1}")
    print(f"R1 (all ratios >= {RATIO_BAR}): {R1_holds}")

    stored_passive_1_7 = next(c["value"] for c in controls if c["cell"] == [1, 7] and c["control"] == "C-passive")
    Qf7 = lab91.haar_frame(7)
    M_born7 = lab101.compute_M_born_gen(Qf7, lab93.U_S)
    z_hist_17 = compute_passive_trajectory(lab93.U_S, lab93.z0, T_NULL)
    znorms_sq_17 = np.sum(np.abs(z_hist_17) ** 2, axis=1)
    D_born_20k = classify_D_born(z_hist_17, znorms_sq_17, Qf7, M_born7, T_TRANSIENT)
    a2_dev = float(abs(D_born_20k - stored_passive_1_7))
    A2 = bool(a2_dev < A2_TOL)
    print(f"A2: D_born(1,7,T=20000)={D_born_20k:.5f} vs stored(T=60000)={stored_passive_1_7:.5f}, "
          f"dev={a2_dev:.5f}, pass={A2}")

    pass_a = bool(A1 and A2)
    stageA = {"ratios": ratios, "R1_holds": R1_holds, "max_disagreement_with_analyst": max_disagreement,
              "regression": {"D_born_1_7_T20k": D_born_20k, "stored_T60k": stored_passive_1_7,
                             "dev": a2_dev, "pass": A2},
              "pass": pass_a}
    return stageA


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("STAGE A - Gates (R1 + A2 regression)")
    print("=" * 60)
    stageA = run_stage_a()
    print(f"\nStage A: {'PASS' if stageA['pass'] else 'FAIL'}")

    if not stageA["pass"]:
        metrics = {
            "lab": "101b", "stream": "cryptographic-substrate", "kind": "hardened-simulation-addendum",
            "stageA": stageA, "stageB": {},
            "verdict": "VOID", "verdict_reason": "Stage A gate(s) failed. Stopping.",
            "leakage_audit": {"wrote_outside_lab_folder": False, "touched_lab101_files": False,
                              "posthoc_threshold_change": False, "metrics_hand_edited": False},
        }
        with open(LAB_FOLDER / "metrics_101b.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nVerdict: VOID")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("STAGE B - Null batteries (300 Haar frames per world)")
    print("=" * 60)
    print("W1 null battery...")
    D_borns_w1, z_hist_w1, znorms_w1 = run_null_battery(lab93.U_S, lab93.z0, W1_SEEDS, T_NULL, T_TRANSIENT)
    print("W2 null battery...")
    D_borns_w2, z_hist_w2, znorms_w2 = run_null_battery(lab101.U_S2, lab101.z0_2, W2_SEEDS, T_NULL, T_TRANSIENT)

    def percentiles(arr):
        pcts = [0, 5, 25, 50, 75, 95, 100]
        vals = np.percentile(arr, pcts)
        return {"min": float(vals[0]), "p5": float(vals[1]), "p25": float(vals[2]), "p50": float(vals[3]),
                "p75": float(vals[4]), "p95": float(vals[5]), "max": float(vals[6])}

    pct_w1 = percentiles(D_borns_w1)
    pct_w2 = percentiles(D_borns_w2)
    print(f"W1 null percentiles: {pct_w1}")
    print(f"W2 null percentiles: {pct_w2}")

    Qf42 = lab91.haar_frame(42)
    M_born42 = lab101.compute_M_born_gen(Qf42, lab93.U_S)
    flagged_w1 = classify_D_born(z_hist_w1, znorms_w1, Qf42, M_born42, T_TRANSIENT)

    Qf23 = lab91.haar_frame(23)
    M_born23 = lab101.compute_M_born_gen(Qf23, lab101.U_S2)
    flagged_w2 = classify_D_born(z_hist_w2, znorms_w2, Qf23, M_born23, T_TRANSIENT)

    placement_w1 = bool(flagged_w1 >= pct_w1["p5"])
    placement_w2 = bool(flagged_w2 >= pct_w2["p5"])
    print(f"\nFlagged W1 frame42 (recomputed T={T_NULL}): {flagged_w1:.5f} vs p5={pct_w1['p5']:.5f}, "
          f"placement={'>= p5 (ordinary)' if placement_w1 else '< p5 (Born-adjacent outlier)'}")
    print(f"Flagged W2 frame23 (recomputed T={T_NULL}): {flagged_w2:.5f} vs p5={pct_w2['p5']:.5f}, "
          f"placement={'>= p5 (ordinary)' if placement_w2 else '< p5 (Born-adjacent outlier)'}")

    R2_holds = bool(placement_w1 and placement_w2)

    # R3 - recorded, not gated
    r3_predicted = pct_w2["p50"] < pct_w1["p50"]
    print(f"\nR3 (recorded): W1 median={pct_w1['p50']:.5f}, W2 median={pct_w2['p50']:.5f}, "
          f"prediction (W2 < W1) {'confirmed' if r3_predicted else 'NOT confirmed'}")

    stageB = {
        "null_percentiles": {"W1": pct_w1, "W2": pct_w2},
        "flagged_placements": {
            "W1_frame42": {"cell": [1, 42], "value_T20k": flagged_w1, "p5": pct_w1["p5"], "placement_ok": placement_w1},
            "W2_frame23": {"cell": [2, 23], "value_T20k": flagged_w2, "p5": pct_w2["p5"], "placement_ok": placement_w2},
        },
        "R2_holds": R2_holds,
        "r3": {"w1_median": pct_w1["p50"], "w2_median": pct_w2["p50"], "prediction_w2_lt_w1_confirmed": r3_predicted},
    }

    print("\nGenerating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (D_borns, pct, flagged, label, world) in zip(
        axes,
        [(D_borns_w1, pct_w1, flagged_w1, "W1 frame42 (flagged)", "W1"),
         (D_borns_w2, pct_w2, flagged_w2, "W2 frame23 (flagged)", "W2")]
    ):
        ax.hist(D_borns, bins=30, color="tab:gray", alpha=0.8, label=f"{world} null (300 frames)")
        ax.axvline(pct["p5"], color="black", linestyle="--", linewidth=1.0, label=f"p5={pct['p5']:.3f}")
        ax.axvline(flagged, color="red", linestyle="-", linewidth=2.0, label=f"{label}={flagged:.3f}")
        ax.set_xlabel("passive D_born (T=20000)")
        ax.set_ylabel("count")
        ax.set_title(f"{world} null distribution")
        ax.legend(fontsize=8)
    fig.suptitle("Lab 101b: passive D_born null distributions vs flagged frames")
    fig.tight_layout()
    fig.savefig(LAB_FOLDER / "nulls_101b.png", dpi=120)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------
    if stageA["R1_holds"] and R2_holds:
        verdict = "CONTROLS-VINDICATED"
        verdict_reason = (f"R1 holds on 6/6 cells (all ratios >= {RATIO_BAR}) AND both flagged cells' "
                           f"recomputed passive values sit at/above their world's 5th percentile "
                           f"(W1 frame42: {flagged_w1:.5f} >= {pct_w1['p5']:.5f}; "
                           f"W2 frame23: {flagged_w2:.5f} >= {pct_w2['p5']:.5f}). "
                           f"The flagged frames are ordinary passive frames, not Born-adjacent outliers.")
    else:
        verdict = "CONTROL-REAL"
        reasons = []
        if not stageA["R1_holds"]:
            failing = [r["cell"] for r in stageA["ratios"] if r["ratio"] < RATIO_BAR]
            reasons.append(f"R1 failed at cells {failing}")
        if not placement_w1:
            reasons.append(f"W1 frame42 ({flagged_w1:.5f}) sits below its world's 5th percentile ({pct_w1['p5']:.5f})")
        if not placement_w2:
            reasons.append(f"W2 frame23 ({flagged_w2:.5f}) sits below its world's 5th percentile ({pct_w2['p5']:.5f})")
        verdict_reason = "; ".join(reasons)

    metrics = {
        "lab": "101b", "stream": "cryptographic-substrate", "kind": "hardened-simulation-addendum",
        "stageA": stageA, "stageB": stageB,
        "verdict": verdict, "verdict_reason": verdict_reason,
        "leakage_audit": {
            "wrote_outside_lab_folder": False, "touched_lab101_files": False,
            "posthoc_threshold_change": False, "metrics_hand_edited": False,
        },
    }
    with open(LAB_FOLDER / "metrics_101b.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")
