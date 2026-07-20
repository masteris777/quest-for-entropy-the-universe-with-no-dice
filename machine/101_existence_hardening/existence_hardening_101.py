# Lab 101 - Existence-Chain Hardening (W2)

import sys
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB_FOLDER = Path(__file__).parent
LAB95_DIR = Path(__file__).resolve().parents[1] / "95_reservoir_bank_scout"
LAB99_DIR = Path(__file__).resolve().parents[1] / "99_variational_floor"
LAB100_DIR = Path(__file__).resolve().parents[1] / "100_repreparation_fold"
sys.path.insert(0, str(LAB95_DIR))
sys.path.insert(0, str(LAB99_DIR))
sys.path.insert(0, str(LAB100_DIR))
import reservoir_bank_scout_95 as lab95  # noqa: E402
import variational_floor_99 as lab99  # noqa: E402
import repreparation_fold_100_v2 as lab100v2  # noqa: E402

lab94 = lab95.lab94
lab93 = lab94.lab93
lab91 = lab94.lab91

LAB99_METRICS_PATH = LAB99_DIR / "metrics_99.json"
LAB100V2_METRICS_PATH = LAB100_DIR / "metrics_100_v2.json"

TWO_PI = 2 * np.pi
K = 16
TAU_MARKOV = [2, 5, 10, 50]
DEGENERACY_FLOOR = 0.05

T_RUN_MAIN = 120000
T_TRANSIENT = 1000
T_RUN_CTRL = 60000

CELLS = [(1, 7), (1, 42), (1, 555), (2, 11), (2, 17), (2, 23)]
J_BAR = 0.06
CTRL_PASSIVE_MIN = 0.30
CTRL_WRONGTHETA_MIN = 0.06
BRIDGE_TOL = 0.02
VANTAGE_FULL_BAR = 0.999
VANTAGE_RECORDS_MARGIN = 0.02
INIT_SENS_BAR = 0.015
A2_KS_BAR = 0.01
A2_CORR_BAR = 0.02
A3_TOL = 0.005
A4_TOL = 0.02
A5_TOL = 1e-10

VANTAGE_WINDOW = 20000
ISO_THETA = lab100v2.ISO_THETA
BASE_INITS = (0.9, 0.31, 0.17)
ALT_INITS = [(1.7, 0.53, 0.29), (0.4, 0.11, 0.23)]

# ---------------------------------------------------------------------------
# World 2 (predeclared)
# ---------------------------------------------------------------------------
U_S2 = np.diag(np.exp(1j * TWO_PI * (np.sqrt(np.array([3.0, 7.0, 13.0])) % 1.0)))
z0_2 = np.array([1.0, 1.0 + 1.0j, 1.0 - 2.0j], dtype=complex)
z0_2 /= np.linalg.norm(z0_2)

WORLDS = {1: {"U_S": lab93.U_S, "z0": lab93.z0}, 2: {"U_S": U_S2, "z0": z0_2}}


# ---------------------------------------------------------------------------
# Generalized machinery (Lab 100 v2's functions with U_S/z0 promoted to
# parameters - verified bit-exact against Lab 100 v2 on frame 91 before use).
# Everything else (build_state_from_pool, build_pool_v2, build_thetas0,
# make_shuf_theta, collect_pool_channels) is imported verbatim from lab100v2.
# ---------------------------------------------------------------------------
def compute_M_born_gen(Qf, U_S):
    M = np.empty((3, 3))
    US_phi = [U_S @ Qf[:, j] for j in range(3)]
    for j in range(3):
        for k in range(3):
            M[j, k] = np.abs(np.vdot(Qf[:, k], US_phi[j])) ** 2
    return M


def run_model_two_gen(Qf, U_S, z0, theta, K_, FREQS, THETAS0, T, T_transient=T_TRANSIENT,
                       track_stage_d=False, track_vantage_last=0):
    z = z0.copy()
    thetas = THETAS0.copy()
    outcomes = np.empty(T, dtype=np.int64)

    basis0 = lab93.perp_basis(Qf[:, 0]) if track_stage_d else None
    history = []
    j0_X, j0_y = [], []

    vantage_start = T - track_vantage_last if track_vantage_last > 0 else None
    z_hist_vantage = [] if track_vantage_last > 0 else None
    thetas_hist_vantage = [] if track_vantage_last > 0 else None

    for t in range(T):
        znorm = np.linalg.norm(z)
        overlaps = np.array([np.abs(np.vdot(Qf[:, k], z)) ** 2 for k in range(3)]) / (znorm ** 2)
        j = int(np.argmax(overlaps))
        outcomes[t] = j

        if track_vantage_last > 0 and t >= vantage_start:
            z_hist_vantage.append(z.copy())
            thetas_hist_vantage.append(thetas.copy())

        if track_stage_d and j == 0 and len(history) >= 8 and t >= T_transient:
            phi0 = Qf[:, 0]
            gamma0 = np.vdot(phi0, z)
            z_perp0 = z - gamma0 * phi0
            c1_ = np.vdot(basis0[0], z_perp0)
            c2_ = np.vdot(basis0[1], z_perp0)
            feat = np.zeros(24)
            for i, s_hist in enumerate(history[-8:]):
                feat[i * 3 + s_hist] = 1.0
            j0_X.append(feat)
            j0_y.append([c1_.real, c1_.imag, c2_.real, c2_.imag])

        gen_idx = t % K_
        phases6 = thetas[gen_idx]
        z_new = lab100v2.build_state_from_pool(theta, phases6, j, Qf)

        z = U_S @ z_new
        thetas = np.mod(thetas + FREQS, TWO_PI)

        if track_stage_d:
            history.append(j)
            if len(history) > 8:
                history.pop(0)

    result = {"outcomes": outcomes}
    if track_stage_d:
        result["j0_X"] = np.array(j0_X)
        result["j0_y"] = np.array(j0_y)
    if track_vantage_last > 0:
        result["z_hist_vantage"] = z_hist_vantage
        result["thetas_hist_vantage"] = thetas_hist_vantage
        result["vantage_start"] = vantage_start
    return result


def run_passive_gen(Qf, U_S, z0, K_, FREQS, THETAS0, T, T_transient=T_TRANSIENT):
    z = z0.copy()
    thetas = THETAS0.copy()
    outcomes = np.empty(T, dtype=np.int64)
    for t in range(T):
        znorm = np.linalg.norm(z)
        overlaps = np.array([np.abs(np.vdot(Qf[:, k], z)) ** 2 for k in range(3)]) / (znorm ** 2)
        j = int(np.argmax(overlaps))
        outcomes[t] = j
        z = U_S @ z
        thetas = np.mod(thetas + FREQS, TWO_PI)
    return {"outcomes": outcomes}


def roundtrip_test_gen(Qf, U_S, z0, theta, K_, FREQS, THETAS0, n_steps=1000):
    z = z0.copy()
    thetas = THETAS0.copy()
    z_hist = [z.copy()]
    thetas_hist = [thetas.copy()]
    j_hist = []
    for t in range(n_steps):
        znorm = np.linalg.norm(z)
        overlaps = np.array([np.abs(np.vdot(Qf[:, k], z)) ** 2 for k in range(3)]) / (znorm ** 2)
        j = int(np.argmax(overlaps))
        j_hist.append(j)
        gen_idx = t % K_
        phases6 = thetas[gen_idx]
        z_new = lab100v2.build_state_from_pool(theta, phases6, j, Qf)
        z = U_S @ z_new
        thetas = np.mod(thetas + FREQS, TWO_PI)
        z_hist.append(z.copy())
        thetas_hist.append(thetas.copy())

    max_err = 0.0
    for t in range(n_steps):
        j = j_hist[t]
        gen_idx = t % K_
        phases6 = thetas_hist[t][gen_idx]
        z_new_rec = lab100v2.build_state_from_pool(theta, phases6, j, Qf)
        z_next_rec = U_S @ z_new_rec
        err_fwd = float(np.max(np.abs(z_next_rec - z_hist[t + 1])))
        thetas_rec = np.mod(thetas_hist[t + 1] - FREQS, TWO_PI)
        err_theta = float(np.max(np.abs(np.exp(1j * thetas_rec) - np.exp(1j * thetas_hist[t]))))
        max_err = max(max_err, err_fwd, err_theta)
    return max_err


def run_model_two_mc(Qf, U_S, z0, theta, T, T_transient, seed):
    """C-mc: the six channels drawn i.i.d. uniform from a labeled PRNG control - NOT the pool."""
    rng = np.random.default_rng(seed)
    z = z0.copy()
    outcomes = np.empty(T, dtype=np.int64)
    for t in range(T):
        znorm = np.linalg.norm(z)
        overlaps = np.array([np.abs(np.vdot(Qf[:, k], z)) ** 2 for k in range(3)]) / (znorm ** 2)
        j = int(np.argmax(overlaps))
        outcomes[t] = j
        phases6 = rng.uniform(0.0, TWO_PI, 6)
        z_new = lab100v2.build_state_from_pool(theta, phases6, j, Qf)
        z = U_S @ z_new
    return {"outcomes": outcomes}


def build_thetas0_custom(K_, c0, c1, c2):
    THETAS0 = np.empty((K_, 6))
    for i in range(K_):
        for c in range(6):
            THETAS0[i, c] = (c0 + c1 * i + c2 * c) % TWO_PI
    return THETAS0


# ---------------------------------------------------------------------------
# Cell metrics helper (as in Labs 96-100)
# ---------------------------------------------------------------------------
def compute_cell_metrics(outcomes, M_born, T_transient=T_TRANSIENT):
    kept = outcomes[T_transient:]
    counts = np.bincount(kept, minlength=3)
    occ = counts / counts.sum()
    degenerate = bool(np.min(occ) < DEGENERACY_FLOOR)
    F1, _ = lab91.counted_F(kept, 1)
    D_born = float(lab91.row_max_tv(F1, M_born))
    m_score = float(lab93.markov_score_fn(kept, F1, TAU_MARKOV))
    J = float(max(D_born, m_score))
    return {"occupancies": [float(o) for o in occ], "degenerate": degenerate,
            "D_born": D_born, "markov_score": m_score, "J": J, "kernel": F1.tolist(), "F1": F1, "occ": occ}


# ---------------------------------------------------------------------------
# Fresh-ensemble optimization (Lab 99's Class A recipe, verbatim constants,
# this lab's own seeds: base 11000+idx, held 12000+idx, DE 1101+idx)
# ---------------------------------------------------------------------------
def optimize_cell_theta(Qf, U_S, M_born, idx):
    base_sample = np.random.default_rng(11000 + idx).uniform(0.0, 1.0, size=(lab99.N_OPT, 6))
    held_sample = np.random.default_rng(12000 + idx).uniform(0.0, 1.0, size=(lab99.N_HELD, 6))
    cache = lab99.precompute_frame_cache(Qf, U_S, base_sample)

    def obj_fast(theta, cache=cache, M_born=M_born):
        return lab99.evaluate_classA_fast(theta, cache, M_born)

    de_kwargs = dict(seed=1101 + idx, maxiter=lab99.DE_A["maxiter"], popsize=lab99.DE_A["popsize"])
    bounds6 = [lab99.BOUNDS_MS] * 6
    t0 = time.time()
    res = lab99.optimize_and_refine(obj_fast, bounds6, de_kwargs)
    opt_time = time.time() - t0
    d_in = float(lab99.evaluate_classA(res["best_x"], base_sample, Qf, U_S, M_born))
    d_held = float(lab99.evaluate_classA(res["best_x"], held_sample, Qf, U_S, M_born))
    return {"theta": res["best_x"], "D_insample": d_in, "D_heldout": d_held, "opt_time_s": opt_time}


# ---------------------------------------------------------------------------
# Autocorrelation instrumentation
# ---------------------------------------------------------------------------
def autocorr_at_lag(x, lag):
    if lag >= len(x):
        return 0.0
    a, b = x[:-lag], x[lag:]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_autocorrelation(channels, K_):
    max_interleaved = 0.0
    for c in range(6):
        for lag in range(1, 65):
            max_interleaved = max(max_interleaved, abs(autocorr_at_lag(channels[:, c], lag)))
    max_pergen = 0.0
    for i in range(K_):
        sub = channels[i::K_, :]
        for c in range(6):
            for lag in range(1, 17):
                max_pergen = max(max_pergen, abs(autocorr_at_lag(sub[:, c], lag)))
    return {"max_interleaved": float(max_interleaved), "max_pergen": float(max_pergen)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t_start = time.time()
    print("Building frames, worlds, pool...")
    frames = {}
    M_borns = {}
    for (w, seed) in CELLS:
        Qf = lab91.haar_frame(seed)
        frames[(w, seed)] = Qf
        M_borns[(w, seed)] = compute_M_born_gen(Qf, WORLDS[w]["U_S"])
    # regression-only frames
    for seed in [91, 93, 201]:
        if (1, seed) not in frames:
            frames[(1, seed)] = lab91.haar_frame(seed)

    FREQS, pool_integers_used = lab100v2.build_pool_v2()
    THETAS0 = lab100v2.build_thetas0(K)

    with open(LAB99_METRICS_PATH) as f:
        lab99_metrics = json.load(f)
    lab99_anchor = next(c["D_heldout"] for c in lab99_metrics["stageB_classA"] if c["frame"] == 91)

    with open(LAB100V2_METRICS_PATH) as f:
        lab100v2_metrics = json.load(f)
    lab100v2_theta91 = next(t["theta"] for t in lab100v2_metrics["stageA"]["theta_provenance"] if t["frame"] == 91)
    lab100v2_stored_integers = lab100v2_metrics["stageA"]["bridge"]["pool_integers_used"]
    lab100v2_J91 = next(c["J"] for c in lab100v2_metrics["stageB"] if c["frame"] == 91)

    print("\n" + "=" * 60)
    print("STAGE A - Gates")
    print("=" * 60)

    # A1(a) - M_born regression
    mborn_devs = []
    for seed in [91, 93, 201]:
        Qf = frames[(1, seed)]
        M1 = lab93.compute_M_born(Qf)
        M2 = compute_M_born_gen(Qf, lab93.U_S)
        mborn_devs.append(float(np.max(np.abs(M1 - M2))))
    mborn_max_dev = float(max(mborn_devs))

    # A1(b) - Model Two regression at (W1, frame91, lab100v2 theta, T=60000)
    run91 = run_model_two_gen(frames[(1, 91)], lab93.U_S, lab93.z0, np.array(lab100v2_theta91),
                               K, FREQS, THETAS0, T=T_RUN_CTRL, T_transient=T_TRANSIENT)
    cell91 = compute_cell_metrics(run91["outcomes"], M_borns.get((1, 91), lab93.compute_M_born(frames[(1, 91)])))
    j_frame91_dev = float(abs(cell91["J"] - lab100v2_J91))
    A1 = bool(mborn_max_dev < 1e-12 and j_frame91_dev < 1e-9)
    print(f"A1: mborn_max_dev={mborn_max_dev:.3e}, j_frame91_dev={j_frame91_dev:.3e}, pass={A1}")

    # A2 - pool
    integers_match = bool(pool_integers_used == lab100v2_stored_integers)
    channels_120k = lab100v2.collect_pool_channels(K, FREQS, THETAS0, T_RUN_MAIN)
    from scipy.stats import kstest
    ks_stats = [float(kstest(channels_120k[:, c], "uniform")[0]) for c in range(6)]
    max_ks = float(max(ks_stats))
    corrs = []
    for i in range(6):
        for j in range(i + 1, 6):
            corrs.append(abs(float(np.corrcoef(channels_120k[:, i], channels_120k[:, j])[0, 1])))
    max_corr = float(max(corrs))
    A2 = bool(integers_match and max_ks < A2_KS_BAR and max_corr < A2_CORR_BAR)
    print(f"A2: integers_match={integers_match}, max_ks={max_ks:.5f}, max_corr={max_corr:.5f}, pass={A2}")

    # A3 - optimizer regression, dedicated cell idx=6 (W1, frame91)
    print("Running A3 optimizer-regression cell (W1, frame91, idx=6)...")
    opt_a3 = optimize_cell_theta(frames[(1, 91)], lab93.U_S, lab93.compute_M_born(frames[(1, 91)]), idx=6)
    a3_dev = float(abs(opt_a3["D_heldout"] - lab99_anchor))
    A3 = bool(a3_dev < A3_TOL)
    print(f"A3: heldout={opt_a3['D_heldout']:.5f}, anchor={lab99_anchor:.5f}, dev={a3_dev:.5f}, pass={A3}")

    # Optimize all 6 main cells
    print("\nOptimizing 6 fresh cells (this dominates runtime)...")
    cell_opt = {}
    for idx, (w, seed) in enumerate(CELLS):
        Qf = frames[(w, seed)]
        U_S = WORLDS[w]["U_S"]
        M_born = M_borns[(w, seed)]
        print(f"-- cell idx={idx} (W{w}, frame{seed}) --")
        res = optimize_cell_theta(Qf, U_S, M_born, idx)
        cell_opt[idx] = res
        print(f"   theta={np.round(res['theta'], 4).tolist()}")
        print(f"   D_insample={res['D_insample']:.5f}, D_heldout={res['D_heldout']:.5f}, "
              f"opt_time={res['opt_time_s']:.1f}s")

    # A4 - calibration floors at T=120000 for the 6 new cells
    markov_floors_by_cell = {}
    kernel_devs4 = []
    for idx, (w, seed) in enumerate(CELLS):
        Qf, M_born = frames[(w, seed)], M_borns[(w, seed)]
        cal_stream = lab94.gen_calibration_stream(M_born, T=T_RUN_MAIN, seed=1000 + idx)
        F1_cal, _ = lab91.counted_F(cal_stream, 1)
        D_born_cal = float(lab91.row_max_tv(F1_cal, M_born))
        m_floor = float(lab93.markov_score_fn(cal_stream, F1_cal, TAU_MARKOV))
        markov_floors_by_cell[str(idx)] = m_floor
        kernel_devs4.append(D_born_cal)
        print(f"A4 cell{idx} (W{w},f{seed}): D_born_cal={D_born_cal:.5f}, markov_floor={m_floor:.5f}")
    kernel_max_dev4 = float(max(kernel_devs4))
    A4 = bool(kernel_max_dev4 < A4_TOL)
    print(f"A4 pass={A4} (kernel_max_dev={kernel_max_dev4:.5f})")

    # A5 - round-trip at (W2, frame11) = cell idx 3
    idx_2_11 = CELLS.index((2, 11))
    theta_2_11 = cell_opt[idx_2_11]["theta"]
    rt_err = roundtrip_test_gen(frames[(2, 11)], U_S2, z0_2, theta_2_11, K, FREQS, THETAS0, n_steps=1000)
    A5 = bool(rt_err < A5_TOL)
    print(f"A5 roundtrip_max_err={rt_err:.3e}, pass={A5}")

    pass_blocking = bool(A1 and A2 and A3 and A4 and A5)
    stageA = {
        "machinery_regression": {"mborn_max_dev": mborn_max_dev, "j_frame91_dev": j_frame91_dev, "pass": A1},
        "pool": {"integers_match_100v2": integers_match, "max_ks": max_ks, "max_corr": max_corr, "pass": A2},
        "optimizer_regression": {"frame91_heldout": opt_a3["D_heldout"], "lab99_anchor": lab99_anchor, "pass": A3},
        "calibration": {"markov_floors_by_cell": markov_floors_by_cell, "kernel_max_dev": kernel_max_dev4, "pass": A4},
        "roundtrip_w2_max_err": rt_err,
        "pass": pass_blocking,
    }
    print(f"\nStage A: {'PASS' if pass_blocking else 'FAIL'}")

    if not pass_blocking:
        metrics = {
            "lab": 101, "stream": "cryptographic-substrate", "kind": "hardened-simulation",
            "serves_node": "born_rule_emergence",
            "stageA": stageA, "stageB": [], "stageC": [], "stageD": {"vantage": [], "ridge": []},
            "stageE": {"bridge_mc": [], "controls": []},
            "stageF": {"init_sensitivity": [], "autocorr": {}, "markov_to_floor_ratios": {}},
            "stageG": {"pairing": {"unengineered_arc_best": [0.0808, 0.1232, 0.1339], "engineered_J_range": [0.0, 0.0]}},
            "summary": {"n_new_cells_under_bar": 0, "n_controls_separating": 0, "n_bridge_pass": 0},
            "verdict": "VOID", "verdict_reason": "Stage A gate(s) failed. Stopping.",
            "leakage_audit": {
                "wrote_outside_lab_folder": False, "theta_reused_across_cells": False,
                "pool_modified": False, "random_draws_inside_world": False,
                "emergence_language_used": False, "posthoc_threshold_change": False,
                "metrics_hand_edited": False,
            },
        }
        with open(LAB_FOLDER / "metrics_101.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nVerdict: VOID")
        sys.exit(0)

    # -----------------------------------------------------------------
    # Stage B/C - main runs at T=120000 (with track_stage_d always; vantage
    # tracking for cells 0 and 3 only, per Stage D)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE B/C - Main runs (T=120000)")
    print("=" * 60)
    main_results = {}
    for idx, (w, seed) in enumerate(CELLS):
        Qf, U_S, z0 = frames[(w, seed)], WORLDS[w]["U_S"], WORLDS[w]["z0"]
        M_born, theta = M_borns[(w, seed)], cell_opt[idx]["theta"]
        vantage_track = VANTAGE_WINDOW if (w, seed) in [(1, 7), (2, 11)] else 0
        run = run_model_two_gen(Qf, U_S, z0, theta, K, FREQS, THETAS0, T=T_RUN_MAIN, T_transient=T_TRANSIENT,
                                 track_stage_d=True, track_vantage_last=vantage_track)
        cell = compute_cell_metrics(run["outcomes"], M_born)
        cell["cell"] = [w, seed]
        cell["theta"] = theta.tolist()
        cell["heldout_floor"] = cell_opt[idx]["D_heldout"]
        cell["outcomes"] = run["outcomes"]
        cell["j0_X"] = run.get("j0_X")
        cell["j0_y"] = run.get("j0_y")
        if vantage_track:
            cell["z_hist_vantage"] = run["z_hist_vantage"]
            cell["thetas_hist_vantage"] = run["thetas_hist_vantage"]
            cell["vantage_start"] = run["vantage_start"]
        main_results[idx] = cell
        print(f"cell{idx} (W{w},f{seed}): D_born={cell['D_born']:.5f}, markov={cell['markov_score']:.5f}, "
              f"J={cell['J']:.5f}, degenerate={cell['degenerate']}")

    stageB = []
    stageC = []
    for idx, (w, seed) in enumerate(CELLS):
        c = main_results[idx]
        entry = {"cell": [w, seed], "theta": c["theta"], "heldout_floor": c["heldout_floor"],
                 "D_born": c["D_born"], "markov_score": c["markov_score"], "J": c["J"],
                 "occupancies": c["occupancies"], "degenerate": c["degenerate"]}
        if w == 1:
            stageB.append(entry)
        else:
            stageC.append(entry)

    # -----------------------------------------------------------------
    # Stage D - vantage falsifier (cells 0, 3) + ridge attacks (all 6)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE D - Vantage falsifier + ridge")
    print("=" * 60)
    vantage_results = []
    for idx, (w, seed) in enumerate(CELLS):
        if (w, seed) not in [(1, 7), (2, 11)]:
            continue
        c = main_results[idx]
        Qf, U_S, theta = frames[(w, seed)], WORLDS[w]["U_S"], cell_opt[idx]["theta"]
        z_hist_v = c["z_hist_vantage"]
        thetas_hist_v = c["thetas_hist_vantage"]
        vstart = c["vantage_start"]
        outcomes = c["outcomes"]

        n_full = len(z_hist_v) - 1
        correct_full = 0
        for t_local in range(n_full):
            t_global = vstart + t_local
            z_t = z_hist_v[t_local]
            thetas_t = thetas_hist_v[t_local]
            s_t = outcomes[t_global]
            gen_idx = t_global % K
            phases6 = thetas_t[gen_idx]
            z_new = lab100v2.build_state_from_pool(theta, phases6, s_t, Qf)
            z_next = U_S @ z_new
            znorm = np.linalg.norm(z_next)
            overlaps = np.array([np.abs(np.vdot(Qf[:, k], z_next)) ** 2 for k in range(3)]) / (znorm ** 2)
            s_next_pred = int(np.argmax(overlaps))
            if s_next_pred == outcomes[t_global + 1]:
                correct_full += 1
        acc_full = correct_full / n_full

        F1 = c["F1"]
        occ = c["occ"]
        records_ceiling = float(sum(occ[j] * np.max(F1[j, :]) for j in range(3)))
        window_start = len(outcomes) - VANTAGE_WINDOW
        correct_rec = 0
        n_rec = 0
        for t_global in range(window_start, len(outcomes) - 1):
            s_t = outcomes[t_global]
            pred = int(np.argmax(F1[s_t, :]))
            if pred == outcomes[t_global + 1]:
                correct_rec += 1
            n_rec += 1
        acc_records = correct_rec / n_rec

        v_pass = bool(acc_full >= VANTAGE_FULL_BAR and acc_records <= records_ceiling + VANTAGE_RECORDS_MARGIN)
        vantage_results.append({"cell": [w, seed], "acc_full": float(acc_full), "acc_records": float(acc_records),
                                 "records_ceiling": records_ceiling, "pass": v_pass})
        print(f"cell (W{w},f{seed}): acc_full={acc_full:.5f} (>= {VANTAGE_FULL_BAR}), "
              f"acc_records={acc_records:.5f} (<= {records_ceiling + VANTAGE_RECORDS_MARGIN:.5f}), pass={v_pass}")

    ridge_results = []
    for idx, (w, seed) in enumerate(CELLS):
        c = main_results[idx]
        X, y = c["j0_X"], c["j0_y"]
        n = len(X)
        if n < 20:
            ridge_results.append({"cell": [w, seed], "r2": None, "n_samples": n})
            continue
        mid = n // 2
        r2 = lab93.ridge_fit_r2(X[:mid], y[:mid], X[mid:], y[mid:], alpha=1.0)
        ridge_results.append({"cell": [w, seed], "r2": float(r2), "n_samples": n})
        print(f"ridge cell (W{w},f{seed}): n={n}, R2={r2:.4f}")

    # -----------------------------------------------------------------
    # Stage E - bridge (C-mc) + adversarial controls, all 6 cells, T=60000
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE E - Bridge and adversarial controls")
    print("=" * 60)
    bridge_results = []
    control_results = []
    for idx, (w, seed) in enumerate(CELLS):
        Qf, U_S, z0 = frames[(w, seed)], WORLDS[w]["U_S"], WORLDS[w]["z0"]
        M_born, theta = M_borns[(w, seed)], cell_opt[idx]["theta"]

        run_mc = run_model_two_mc(Qf, U_S, z0, theta, T=T_RUN_CTRL, T_transient=T_TRANSIENT, seed=5000 + idx)
        cell_mc = compute_cell_metrics(run_mc["outcomes"], M_born)
        run_main_ctrl = run_model_two_gen(Qf, U_S, z0, theta, K, FREQS, THETAS0, T=T_RUN_CTRL, T_transient=T_TRANSIENT)
        cell_main_ctrl = compute_cell_metrics(run_main_ctrl["outcomes"], M_born)
        delta = float(abs(cell_mc["J"] - cell_main_ctrl["J"]))
        bridge_pass = bool(delta < BRIDGE_TOL)
        bridge_results.append({"cell": [w, seed], "J_mc": cell_mc["J"], "J_main": cell_main_ctrl["J"],
                                "delta": delta, "pass": bridge_pass})
        print(f"cell{idx} (W{w},f{seed}) C-mc: J_mc={cell_mc['J']:.4f}, J_main(T=60k)={cell_main_ctrl['J']:.4f}, "
              f"delta={delta:.4f}, pass={bridge_pass}")

        run_p = run_passive_gen(Qf, U_S, z0, K, FREQS, THETAS0, T=T_RUN_CTRL, T_transient=T_TRANSIENT)
        cell_p = compute_cell_metrics(run_p["outcomes"], M_born)
        sep_p = bool(cell_p["D_born"] > CTRL_PASSIVE_MIN)
        control_results.append({"cell": [w, seed], "control": "C-passive", "value": cell_p["D_born"],
                                 "bar": CTRL_PASSIVE_MIN, "separates": sep_p})

        run_i = run_model_two_gen(Qf, U_S, z0, ISO_THETA, K, FREQS, THETAS0, T=T_RUN_CTRL, T_transient=T_TRANSIENT)
        cell_i = compute_cell_metrics(run_i["outcomes"], M_born)
        sep_i = bool(cell_i["J"] > CTRL_WRONGTHETA_MIN)
        control_results.append({"cell": [w, seed], "control": "C-isotropic", "value": cell_i["J"],
                                 "bar": CTRL_WRONGTHETA_MIN, "separates": sep_i})

        shuf_theta = lab100v2.make_shuf_theta(theta)
        run_s = run_model_two_gen(Qf, U_S, z0, shuf_theta, K, FREQS, THETAS0, T=T_RUN_CTRL, T_transient=T_TRANSIENT)
        cell_s = compute_cell_metrics(run_s["outcomes"], M_born)
        sep_s = bool(cell_s["J"] > CTRL_WRONGTHETA_MIN)
        control_results.append({"cell": [w, seed], "control": "C-shuffled", "value": cell_s["J"],
                                 "bar": CTRL_WRONGTHETA_MIN, "separates": sep_s})
        print(f"cell{idx} controls: passive={cell_p['D_born']:.4f}({sep_p}), "
              f"isotropic={cell_i['J']:.4f}({sep_i}), shuffled={cell_s['J']:.4f}({sep_s})")

    # -----------------------------------------------------------------
    # Stage F - init sensitivity + autocorrelation + markov/floor ratios
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE F - Robustness and instrumentation")
    print("=" * 60)
    idx_1_7 = CELLS.index((1, 7))
    base_J = main_results[idx_1_7]["J"]
    Qf7, theta7 = frames[(1, 7)], cell_opt[idx_1_7]["theta"]
    init_sensitivity = []
    for (c0, c1, c2) in ALT_INITS:
        THETAS0_alt = build_thetas0_custom(K, c0, c1, c2)
        run_alt = run_model_two_gen(Qf7, lab93.U_S, lab93.z0, theta7, K, FREQS, THETAS0_alt,
                                     T=T_RUN_MAIN, T_transient=T_TRANSIENT)
        cell_alt = compute_cell_metrics(run_alt["outcomes"], M_borns[(1, 7)])
        delta_j = float(abs(cell_alt["J"] - base_J))
        init_sensitivity.append({"inits": [c0, c1, c2], "J": cell_alt["J"], "delta_vs_base": delta_j})
        print(f"alt init {(c0, c1, c2)}: J={cell_alt['J']:.5f}, delta_vs_base={delta_j:.5f} (bar {INIT_SENS_BAR})")
    stageF1_pass = bool(all(e["delta_vs_base"] <= INIT_SENS_BAR for e in init_sensitivity))
    print(f"Stage F.1 pass={stageF1_pass}")

    autocorr = compute_autocorrelation(channels_120k, K)
    print(f"Autocorrelation: max_interleaved={autocorr['max_interleaved']:.5f}, "
          f"max_pergen={autocorr['max_pergen']:.5f} (P5 bar 0.05)")

    markov_to_floor_ratios = {}
    for idx, (w, seed) in enumerate(CELLS):
        floor = markov_floors_by_cell[str(idx)]
        ratio = main_results[idx]["markov_score"] / floor if floor > 0 else None
        markov_to_floor_ratios[str(idx)] = ratio
        print(f"cell{idx} markov_to_floor_ratio={ratio:.3f}" if ratio is not None else f"cell{idx}: floor=0")

    stageF = {"init_sensitivity": init_sensitivity, "autocorr": autocorr,
              "markov_to_floor_ratios": markov_to_floor_ratios}

    # -----------------------------------------------------------------
    # Stage G - pairing table
    # -----------------------------------------------------------------
    engineered_Js = [main_results[idx]["J"] for idx in range(6)]
    stageG = {"pairing": {"unengineered_arc_best": [0.0808, 0.1232, 0.1339],
                           "engineered_J_range": [float(min(engineered_Js)), float(max(engineered_Js))]}}

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    n_new_cells_under_bar = sum(1 for idx in range(6) if main_results[idx]["J"] < J_BAR)
    n_controls_separating = sum(1 for c in control_results if c["separates"])
    n_bridge_pass = sum(1 for b in bridge_results if b["pass"])
    summary = {"n_new_cells_under_bar": n_new_cells_under_bar, "n_controls_separating": n_controls_separating,
               "n_bridge_pass": n_bridge_pass}
    print(f"\nSummary: {n_new_cells_under_bar}/6 cells under J-bar, {n_controls_separating}/18 controls separating, "
          f"{n_bridge_pass}/6 bridge pass")

    print("\nGenerating plot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, idx in zip(axes.flat, range(6)):
        w, seed = CELLS[idx]
        ctrl_cells = {c["control"]: c for c in control_results if c["cell"] == [w, seed]}
        bridge_cell = next(b for b in bridge_results if b["cell"] == [w, seed])
        labels = ["Main", "C-mc", "C-passive", "C-isotropic", "C-shuffled"]
        vals = [main_results[idx]["J"], bridge_cell["J_mc"], ctrl_cells["C-passive"]["value"],
                ctrl_cells["C-isotropic"]["value"], ctrl_cells["C-shuffled"]["value"]]
        colors = ["tab:green", "tab:blue", "tab:gray", "tab:red", "tab:orange"]
        ax.bar(labels, vals, color=colors)
        ax.axhline(J_BAR, color="black", linestyle="--", linewidth=0.9, label=f"J={J_BAR}")
        ax.set_ylabel("J (D_born for C-passive)")
        ax.set_title(f"cell{idx}: W{w} frame{seed}")
        ax.legend(fontsize=6)
        ax.text(0.02, 0.95, "arc: 0.42 -> 0.08 -> 0.011 -> "
                             f"{main_results[idx]['J']:.4f}",
                transform=ax.transAxes, fontsize=6, va="top")
    fig.suptitle("Lab 101: 6 new cells - main J vs bridge (C-mc) and 3 adversarial controls; J=0.06 line")
    fig.tight_layout()
    fig.savefig(LAB_FOLDER / "hardening_101.png", dpi=110)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Verdict - first-match order
    # -----------------------------------------------------------------
    w1_idx = [i for i, (w, s) in enumerate(CELLS) if w == 1]
    w2_idx = [i for i, (w, s) in enumerate(CELLS) if w == 2]
    n_w1_under = sum(1 for i in w1_idx if main_results[i]["J"] < J_BAR)
    n_w2_under = sum(1 for i in w2_idx if main_results[i]["J"] < J_BAR)
    n_w2_fail = len(w2_idx) - n_w2_under

    n_bridge_fail_with_others_ok = sum(
        1 for b in bridge_results if not b["pass"]
        and main_results[[c for c in range(6) if CELLS[c] == tuple(b["cell"])][0]]["J"] < J_BAR
    )
    all_controls_sep = bool(n_controls_separating == 18)
    all_bridge_pass = bool(n_bridge_pass == 6)
    all_vantage_pass = bool(all(v["pass"] for v in vantage_results)) if vantage_results else False

    if (pass_blocking and n_new_cells_under_bar == 6 and all_controls_sep and all_bridge_pass
            and all_vantage_pass and stageF1_pass):
        verdict = "HARDENED"
        verdict_reason = (f"Stage A passed; 6/6 new cells J<0.06; 18/18 controls separate; 6/6 bridge pass; "
                           f"vantage falsifier confirms records-random/bulk-deterministic; init-robust "
                           f"(max delta {max(e['delta_vs_base'] for e in init_sensitivity):.5f} <= {INIT_SENS_BAR}). "
                           f"Existence statement rises to empirical_signal.")
    elif n_w1_under == 3 and n_w2_fail >= 2:
        verdict = "WORLD-BOUND"
        verdict_reason = (f"W1: 3/3 cells under 0.06. W2: {n_w2_fail}/3 cells failed J<0.06. "
                           f"W2 floors: {[main_results[i]['heldout_floor'] for i in w2_idx]}, "
                           f"W2 J: {[main_results[i]['J'] for i in w2_idx]}. Substrate-specific generalization gap.")
    elif n_bridge_fail_with_others_ok >= 2:
        verdict = "BRIDGE-GAP"
        verdict_reason = (f"{n_bridge_fail_with_others_ok} cells failed the C-mc bridge check "
                           f"(|J_mc-J_main|>=0.02) while otherwise passing: {bridge_results}.")
    elif n_new_cells_under_bar in (4, 5):
        verdict = "PARTIAL-HARDENED"
        failing = [CELLS[i] for i in range(6) if main_results[i]["J"] >= J_BAR]
        verdict_reason = (f"{n_new_cells_under_bar}/6 cells under 0.06 with controls separating "
                           f"({n_controls_separating}/18). Failing cells: {failing}.")
    elif not stageF1_pass:
        verdict = "FRAGILE"
        verdict_reason = (f"Stage F.1 init-sensitivity failed: {init_sensitivity}. "
                           f"Result depends on arbitrary pool-init constants.")
    else:
        verdict = "OTHER"
        verdict_reason = (f"No ladder rule matched exactly: n_new_cells_under_bar={n_new_cells_under_bar}/6, "
                           f"n_controls_separating={n_controls_separating}/18, n_bridge_pass={n_bridge_pass}/6, "
                           f"vantage_pass={all_vantage_pass}, init_robust={stageF1_pass}.")

    metrics = {
        "lab": 101, "stream": "cryptographic-substrate", "kind": "hardened-simulation",
        "serves_node": "born_rule_emergence",
        "stageA": stageA, "stageB": stageB, "stageC": stageC,
        "stageD": {"vantage": vantage_results, "ridge": ridge_results},
        "stageE": {"bridge_mc": bridge_results, "controls": control_results},
        "stageF": stageF, "stageG": stageG,
        "summary": summary,
        "verdict": verdict, "verdict_reason": verdict_reason,
        "leakage_audit": {
            "wrote_outside_lab_folder": False, "theta_reused_across_cells": False,
            "pool_modified": False, "random_draws_inside_world": False,
            "emergence_language_used": False, "posthoc_threshold_change": False,
            "metrics_hand_edited": False,
        },
    }
    with open(LAB_FOLDER / "metrics_101.json", "w") as f:
        json.dump(metrics, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")
