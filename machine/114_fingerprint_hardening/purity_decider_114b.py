# Lab 114b - The Purity Micro-Decider
#
# Completes the Lab 114 battery: 91-P2-asym failed purity alone (R=0.1304 vs 0.10),
# entirely on weak channel k=2, which carries the same absolute overtone power as the
# two passing channels but divides it by ~180x less fringe power. Quadruple N and let
# the numbers decide whether that ratio was noise (falls ~4x) or structure (holds).

import sys
import json
import time
from pathlib import Path

import numpy as np

LAB_FOLDER = Path(__file__).parent
sys.path.insert(0, str(LAB_FOLDER))
import fingerprint_hardening_114 as lab114  # noqa: E402

lab99 = lab114.lab99
lab110 = lab114.lab110
lab91 = lab114.lab91
lab93 = lab114.lab93

PHI_GRID = lab114.PHI_GRID
N_TIER1 = lab114.N_TIER1  # 200,000
N_DECIDER = 800_000
RMS_BAR_W1 = lab114.RMS_BAR_W1
A1_ANCHOR_TOL = 1e-9
OVERTONE_BAR = lab114.OVERTONE_BAR  # 0.10
POWER_NOISE_MULT = 2.0
POWER_STRUCTURE_FLOOR = 0.0115

if __name__ == "__main__":
    t_start = time.time()

    print("=" * 60)
    print("STAGE A0 - Reconstruct theta manifest and target vectors (verbatim, run-2 path)")
    print("=" * 60)
    with open(lab114.LAB99_DIR / "metrics_99.json") as f:
        m99 = json.load(f)
    class_a = {c["frame"]: c for c in m99["stageB_classA"]}
    theta_91 = np.array(class_a[91]["theta"])
    theta_93 = np.array(class_a[93]["theta"])

    Qf91 = lab91.haar_frame(91)
    Qf93 = lab91.haar_frame(93)
    U_S = lab93.U_S
    uA91, uB91 = Qf91[:, 0], Qf91[:, 1]
    uA93, uB93 = Qf93[:, 0], Qf93[:, 1]
    uB91_P2 = (Qf91[:, 1] + Qf91[:, 2]) / np.sqrt(2)
    uB93_P2 = (Qf93[:, 1] + Qf93[:, 2]) / np.sqrt(2)

    with open(LAB_FOLDER / "metrics_114.json") as f:
        m114 = json.load(f)
    src_cell = m114["stageB"]["91-P2-asym"]
    src_F = np.array(src_cell["F"])

    print("\n" + "=" * 60)
    print("STAGE A - Anchor (else VOID)")
    print("=" * 60)
    # Run 2's exact seed for 91-P2-asym: cells_plan index 4, idx offset = 24*4 = 96 -> base 16096.
    seed_base_anchor = 16096
    F_anchor = np.empty((24, 3))
    for i, phi in enumerate(PHI_GRID):
        F_anchor[i] = lab114.counted_F_engineered(theta_91, "asym", uA91, uB91_P2, Qf91, U_S, phi,
                                                   N_TIER1, seed_base_anchor + i)
    anchor_max_diff = float(np.max(np.abs(F_anchor - src_F)))
    stageA_pass = bool(anchor_max_diff < A1_ANCHOR_TOL)
    print(f"  seed base={seed_base_anchor}, N={N_TIER1}: max|F_anchor - stored| = {anchor_max_diff:.3e} "
          f"(<{A1_ANCHOR_TOL}), pass={stageA_pass}")

    if not stageA_pass:
        metrics = {
            "lab": "114b", "stream": "cryptographic-substrate", "kind": "hardened-simulation-micro",
            "parent": 114, "serves_node": "born_rule_emergence",
            "stageA": {"anchor_max_diff": anchor_max_diff, "anchor_seed_base": seed_base_anchor, "pass": False},
            "stageB": {}, "stageC": {},
            "verdict": "VOID", "verdict_reason": f"Stage A anchor failed: max diff {anchor_max_diff:.3e} >= {A1_ANCHOR_TOL}.",
            "leakage_audit": {
                "wrote_outside_lab_folder": False, "purity_definition_changed": False,
                "theta_adjusted": False, "seed_bar_N_changed": False, "posthoc_threshold_change": False,
                "metrics_hand_edited": False, "engineering_declared": True,
            },
        }
        with open(LAB_FOLDER / "metrics_114b.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nVerdict: VOID")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("STAGE B - The decider (91-P2-asym at N=800,000)")
    print("=" * 60)
    seed_base_decider = 17000
    F_800k = np.empty((24, 3))
    for i, phi in enumerate(PHI_GRID):
        F_800k[i] = lab114.counted_F_engineered(theta_91, "asym", uA91, uB91_P2, Qf91, U_S, phi,
                                                 N_DECIDER, seed_base_decider + i)
    P_all_91P2 = np.array([lab114.analytic_P("asym", uA91, uB91_P2, Qf91, U_S, phi) for phi in PHI_GRID])
    cell_800k = lab114.assemble_cell(F_800k, P_all_91P2, RMS_BAR_W1)
    cell_800k["kind_of_cell"] = "engineered"
    cell_800k["frame"] = 91

    print(f"  RMS={cell_800k['rms_overall']:.4f} (bar {RMS_BAR_W1}), max_active_R={cell_800k['max_active_R']:.5f}, "
          f"S_chosen={cell_800k['S_chosen']}, shape_match={cell_800k['shape_match']}, "
          f"purity_ok={cell_800k['purity_ok']}, S_ok={cell_800k['S_ok']}")
    for k in range(3):
        o = cell_800k["overtone_per_k"][k]
        print(f"  channel k={k}: h1_power={o['h1_power']:.5f}, overtone_power={o['overtone_power']:.5f}, "
              f"R={o['R']:.5f}")

    # Noise floor at 800k: twin pair at phi=0, seeds 17200/17201
    Pa = lab114.counted_F_engineered(theta_91, "asym", uA91, uB91_P2, Qf91, U_S, 0.0, N_DECIDER, 17200)
    Pb = lab114.counted_F_engineered(theta_91, "asym", uA91, uB91_P2, Qf91, U_S, 0.0, N_DECIDER, 17201)
    noise_floor_800k = float(np.max(np.abs(Pa - Pb)))
    print(f"\n  Noise floor (twin pair, seeds 17200/17201, phi=0): {noise_floor_800k:.5f}")

    # --- The frozen fork ---
    R_k2 = cell_800k["overtone_per_k"][2]["R"]
    power_k2 = cell_800k["overtone_per_k"][2]["overtone_power"]
    power_k0 = cell_800k["overtone_per_k"][0]["overtone_power"]
    power_k1 = cell_800k["overtone_per_k"][1]["overtone_power"]
    mean_power_strong = (power_k0 + power_k1) / 2.0

    clause_artifact = bool(R_k2 <= OVERTONE_BAR and power_k2 <= POWER_NOISE_MULT * mean_power_strong)
    clause_structure = bool(R_k2 > OVERTONE_BAR and power_k2 >= POWER_STRUCTURE_FLOOR)

    if clause_artifact:
        fork = "ARTIFACT"
    elif clause_structure:
        fork = "STRUCTURE"
    else:
        fork = "OTHER"

    print(f"\n  FORK: R_k2={R_k2:.5f} (bar {OVERTONE_BAR}), power_k2={power_k2:.5f}, "
          f"mean(power_k0,power_k1)={mean_power_strong:.5f} (2x = {2*mean_power_strong:.5f}), "
          f"structure floor={POWER_STRUCTURE_FLOOR}")
    print(f"  ARTIFACT clause (R<=0.10 AND power_k2<=2x mean strong): {clause_artifact}")
    print(f"  STRUCTURE clause (R>0.10 AND power_k2>=0.0115): {clause_structure}")
    print(f"  Fork result: {fork}")

    fork_eval = {
        "R_k2_800k": R_k2, "power_k2_800k": power_k2, "power_k0_800k": power_k0, "power_k1_800k": power_k1,
        "mean_power_strong_800k": mean_power_strong, "artifact_power_bound": 2 * mean_power_strong,
        "structure_power_floor": POWER_STRUCTURE_FLOOR, "overtone_bar": OVERTONE_BAR,
        "clause_artifact": clause_artifact, "clause_structure": clause_structure, "fork": fork,
    }

    # Run-2 vs 800k side-by-side (for the report)
    run2_overtone = m114["stageB"]["91-P2-asym"]["overtone_per_k"]
    side_by_side = {
        "run2_N": N_TIER1, "decider_N": N_DECIDER,
        "run2": {"R_per_k": [o["R"] for o in run2_overtone],
                 "overtone_power_per_k": [o["overtone_power"] for o in run2_overtone],
                 "h1_power_per_k": [o["h1_power"] for o in run2_overtone],
                 "rms_overall": m114["stageB"]["91-P2-asym"]["rms_overall"],
                 "S_chosen": m114["stageB"]["91-P2-asym"]["S_chosen"]},
        "decider_800k": {"R_per_k": [o["R"] for o in cell_800k["overtone_per_k"]],
                          "overtone_power_per_k": [o["overtone_power"] for o in cell_800k["overtone_per_k"]],
                          "h1_power_per_k": [o["h1_power"] for o in cell_800k["overtone_per_k"]],
                          "rms_overall": cell_800k["rms_overall"],
                          "S_chosen": cell_800k["S_chosen"]},
        "predicted": {"power_k2": 0.0058, "R_k2": 0.033},
    }
    print("\n  Run 2 (N=200k) vs Decider (N=800k) side by side:")
    print(f"    R per k       : run2={side_by_side['run2']['R_per_k']} -> 800k={side_by_side['decider_800k']['R_per_k']}")
    print(f"    overtone power: run2={side_by_side['run2']['overtone_power_per_k']} -> 800k={side_by_side['decider_800k']['overtone_power_per_k']}")
    print(f"    RMS           : run2={side_by_side['run2']['rms_overall']:.4f} -> 800k={side_by_side['decider_800k']['rms_overall']:.4f}")
    print(f"    S_chosen      : run2={side_by_side['run2']['S_chosen']} -> 800k={side_by_side['decider_800k']['S_chosen']}")

    print("\n" + "=" * 60)
    print("STAGE C - Breadth (93-P2-asym at N=200,000, measurement-only, excluded from verdict)")
    print("=" * 60)
    seed_base_breadth = 17100
    F_93P2 = np.empty((24, 3))
    for i, phi in enumerate(PHI_GRID):
        F_93P2[i] = lab114.counted_F_engineered(theta_93, "asym", uA93, uB93_P2, Qf93, U_S, phi,
                                                 N_TIER1, seed_base_breadth + i)
    P_all_93P2 = np.array([lab114.analytic_P("asym", uA93, uB93_P2, Qf93, U_S, phi) for phi in PHI_GRID])
    cell_93P2 = lab114.assemble_cell(F_93P2, P_all_93P2, RMS_BAR_W1)
    cell_93P2["kind_of_cell"] = "engineered"
    cell_93P2["frame"] = 93
    print(f"  93-P2-asym: RMS={cell_93P2['rms_overall']:.4f} (bar {RMS_BAR_W1}), "
          f"max_active_R={cell_93P2['max_active_R']:.5f}, S_chosen={cell_93P2['S_chosen']}, "
          f"shape_match={cell_93P2['shape_match']}, purity_ok={cell_93P2['purity_ok']}, S_ok={cell_93P2['S_ok']}")
    print("  (measurement-only - does NOT enter the verdict)")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if fork == "ARTIFACT":
        verdict = "ARTIFACT-CONFIRMED"
        verdict_reason = (f"R_k2(800k)={R_k2:.5f}<=0.10 and power_k2={power_k2:.5f}<=2x mean strong "
                           f"({2*mean_power_strong:.5f}) - noise-consistency clause satisfied. Battery completes 6/6.")
    elif fork == "STRUCTURE":
        verdict = "STRUCTURE-CONFIRMED"
        verdict_reason = (f"R_k2(800k)={R_k2:.5f}>0.10 and power_k2={power_k2:.5f}>=0.0115 - the ratio did not "
                           f"scale away with N. Real composite-target overtone content; bounded scope note.")
    else:
        verdict = "OTHER"
        verdict_reason = (f"Neither fork clause matched cleanly: R_k2={R_k2:.5f}, power_k2={power_k2:.5f}, "
                           f"mean_power_strong={mean_power_strong:.5f}. Reported straight, no forcing.")
    print(f"Verdict: {verdict}")
    print(f"Reason: {verdict_reason}")

    def strip_for_json(d):
        return json.loads(json.dumps(d, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))

    metrics = {
        "lab": "114b", "stream": "cryptographic-substrate", "kind": "hardened-simulation-micro",
        "parent": 114, "serves_node": "born_rule_emergence",
        "stageA": {"anchor_max_diff": anchor_max_diff, "anchor_seed_base": seed_base_anchor,
                   "anchor_N": N_TIER1, "pass": stageA_pass},
        "stageB": strip_for_json({"cell_800k": cell_800k, "noise_floor_800k": noise_floor_800k,
                                   "fork_eval": fork_eval, "side_by_side": side_by_side,
                                   "seed_base": seed_base_decider, "N": N_DECIDER}),
        "stageC": strip_for_json({"93-P2-asym": cell_93P2, "seed_base": seed_base_breadth, "N": N_TIER1,
                                   "note": "measurement-only, predeclared excluded from verdict"}),
        "verdict": verdict, "verdict_reason": verdict_reason,
        "leakage_audit": {
            "wrote_outside_lab_folder": False, "purity_definition_changed": False,
            "theta_adjusted": False, "seed_bar_N_changed": False, "posthoc_threshold_change": False,
            "metrics_hand_edited": False, "engineering_declared": True,
        },
    }
    with open(LAB_FOLDER / "metrics_114b.json", "w") as f:
        json.dump(metrics, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")
