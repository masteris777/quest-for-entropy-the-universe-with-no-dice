# Lab 100 v2 - The Re-Preparation Fold (clean-pool rerun)
#
# = repreparation_fold_100.py (v1) with EXACTLY ONE functional change: the pool
# frequency builder (build_pool -> build_pool_v2, per lab-goal-v2.md). Every
# other element - Model Two's step, thetas from metrics_99.json stageB_classA,
# gates A1/A3/A4, Stage B/C/D, the verdict ladder, the bars - is unchanged from v1.

import sys
import json
from pathlib import Path

import numpy as np
from scipy.stats import kstest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB_FOLDER = Path(__file__).parent
LAB95_DIR = Path(__file__).resolve().parents[1] / "95_reservoir_bank_scout"
LAB99_DIR = Path(__file__).resolve().parents[1] / "99_variational_floor"
sys.path.insert(0, str(LAB95_DIR))
import reservoir_bank_scout_95 as lab95  # noqa: E402

lab94 = lab95.lab94
lab93 = lab94.lab93
lab91 = lab94.lab91

LAB99_METRICS_PATH = LAB99_DIR / "metrics_99.json"

# ---------------------------------------------------------------------------
# Predeclared parameters (unchanged from v1)
# ---------------------------------------------------------------------------
K = 16
T_RUN = 60000
T_TRANSIENT = 1000
FRAME_SEEDS = [91, 93, 201]
TAU_MARKOV = [2, 5, 10, 50]
DEGENERACY_FLOOR = 0.05
CAL_SEED = 1000
J_BAR = 0.06
CTRL_PASSIVE_MIN = 0.30
CTRL_WRONGTHETA_MIN = 0.06
ISO_THETA = np.array([0.5774, 0.5, 0.5774, 0.5, 0.5774, 0.5])
A2_KS_BAR = 0.01
A2_CORR_BAR = 0.02
A4_TOL = 1e-10

OTHER_IDX = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
TWO_PI = 2 * np.pi

# v2 pool-builder margins (predeclared in lab-goal-v2.md, binding)
MARGIN_1 = 0.3
MARGIN_2 = 0.1


# ---------------------------------------------------------------------------
# The pool v2 - the ONLY functional change vs v1. Exact code per lab-goal-v2.md.
# Frequencies drawn from square-free integers (square roots of distinct
# square-free integers are linearly independent over Q - no exact resonance of
# any order), with a greedy within-generator near-resonance screen on aliased
# values a = (K*f) % 2*pi: first-order margin 0.3, second-order margin 0.1.
# ---------------------------------------------------------------------------
def is_squarefree(n):
    d = 2
    while d * d <= n:
        if n % (d * d) == 0:
            return False
        d += 1
    return True


def squarefree_gen(start=2):
    n = start
    while True:
        if is_squarefree(n):
            yield n
        n += 1


def circ(x):
    d = abs(x) % TWO_PI
    return min(d, TWO_PI - d)


def build_pool_v2():
    gen = squarefree_gen()
    FREQS = np.empty((K, 6))
    used = []
    for i in range(K):
        acc = []  # aliased alphas accepted in this generator
        for c in range(6):
            while True:
                n = next(gen)
                f = TWO_PI * (np.sqrt(float(n)) % 1.0)
                a = (K * f) % TWO_PI
                ok = all(
                    circ(a - b) > MARGIN_1 and circ(a + b) > MARGIN_1
                    and circ(2 * a - b) > MARGIN_2 and circ(2 * a + b) > MARGIN_2
                    and circ(a - 2 * b) > MARGIN_2 and circ(a + 2 * b) > MARGIN_2
                    for b in acc
                )
                if ok:
                    FREQS[i, c] = f
                    acc.append(a)
                    used.append(n)
                    break
    return FREQS, used


def build_thetas0(K_):
    # Inits UNCHANGED from v1
    THETAS0 = np.empty((K_, 6))
    for i in range(K_):
        for c in range(6):
            THETAS0[i, c] = (0.9 + 0.31 * i + 0.17 * c) % TWO_PI
    return THETAS0


def make_shuf_theta(theta):
    m_g, s_g, m_1, s_1, m_2, s_2 = theta
    return np.array([m_1, s_1, m_g, s_g, m_2, s_2])


def build_state_from_pool(theta, phases6, j, Qf):
    m_g, s_g, m_1, s_1, m_2, s_2 = theta
    v0, v1, v2, w0, w1, w2 = phases6 / TWO_PI
    g_mag = max(0.0, m_g + s_g * (v0 - 0.5))
    c1 = max(0.0, m_1 + s_1 * (v1 - 0.5))
    c2 = max(0.0, m_2 + s_2 * (v2 - 0.5))
    k, l = OTHER_IDX[j]
    phi_j, phi_k, phi_l = Qf[:, j], Qf[:, k], Qf[:, l]
    z = (g_mag * np.exp(2j * np.pi * w0) * phi_j
         + c1 * np.exp(2j * np.pi * w1) * phi_k
         + c2 * np.exp(2j * np.pi * w2) * phi_l)
    return z


# ---------------------------------------------------------------------------
# Model Two step loop (unchanged from v1). Maximal fold: the ENTIRE old state
# z_t is displaced to the tape; the new state is drawn PURELY from the pool in
# the outcome's own basis.
# ---------------------------------------------------------------------------
def run_model_two(Qf, theta, K_, FREQS, THETAS0, T, T_transient=T_TRANSIENT, track_stage_d=False):
    z = lab93.z0.copy()
    thetas = THETAS0.copy()
    outcomes = np.empty(T, dtype=np.int64)

    basis0 = lab93.perp_basis(Qf[:, 0]) if track_stage_d else None
    history = []
    j0_X, j0_y = [], []

    for t in range(T):
        znorm = np.linalg.norm(z)
        overlaps = np.array([np.abs(np.vdot(Qf[:, k], z)) ** 2 for k in range(3)]) / (znorm ** 2)
        j = int(np.argmax(overlaps))
        outcomes[t] = j

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
        z_new = build_state_from_pool(theta, phases6, j, Qf)  # maximal fold: z_t discarded (to tape), rebuilt from pool

        z = lab93.U_S @ z_new
        thetas = np.mod(thetas + FREQS, TWO_PI)

        if track_stage_d:
            history.append(j)
            if len(history) > 8:
                history.pop(0)

    result = {"outcomes": outcomes}
    if track_stage_d:
        result["j0_X"] = np.array(j0_X)
        result["j0_y"] = np.array(j0_y)
    return result


def run_passive(Qf, K_, FREQS, THETAS0, T, T_transient=T_TRANSIENT):
    z = lab93.z0.copy()
    thetas = THETAS0.copy()
    outcomes = np.empty(T, dtype=np.int64)
    for t in range(T):
        znorm = np.linalg.norm(z)
        overlaps = np.array([np.abs(np.vdot(Qf[:, k], z)) ** 2 for k in range(3)]) / (znorm ** 2)
        j = int(np.argmax(overlaps))
        outcomes[t] = j
        z = lab93.U_S @ z  # no fold: pure substrate evolution
        thetas = np.mod(thetas + FREQS, TWO_PI)  # pool ticks for parity; unused
    return {"outcomes": outcomes}


def collect_pool_channels(K_, FREQS, THETAS0, T):
    thetas = THETAS0.copy()
    channels = np.empty((T, 6))
    for t in range(T):
        gen_idx = t % K_
        channels[t, :] = thetas[gen_idx] / TWO_PI
        thetas = np.mod(thetas + FREQS, TWO_PI)
    return channels


def roundtrip_test(Qf, theta, K_, FREQS, THETAS0, n_steps=1000):
    z = lab93.z0.copy()
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
        z_new = build_state_from_pool(theta, phases6, j, Qf)
        z = lab93.U_S @ z_new
        thetas = np.mod(thetas + FREQS, TWO_PI)
        z_hist.append(z.copy())
        thetas_hist.append(thetas.copy())

    max_err = 0.0
    for t in range(n_steps):
        j = j_hist[t]
        gen_idx = t % K_
        phases6 = thetas_hist[t][gen_idx]
        z_new_rec = build_state_from_pool(theta, phases6, j, Qf)
        z_next_rec = lab93.U_S @ z_new_rec
        err_fwd = float(np.max(np.abs(z_next_rec - z_hist[t + 1])))

        thetas_rec = np.mod(thetas_hist[t + 1] - FREQS, TWO_PI)
        err_theta = float(np.max(np.abs(np.exp(1j * thetas_rec) - np.exp(1j * thetas_hist[t]))))

        max_err = max(max_err, err_fwd, err_theta)
    return max_err


# ---------------------------------------------------------------------------
# Metrics helper (unchanged from v1)
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
            "D_born": D_born, "markov_score": m_score, "J": J, "kernel": F1.tolist()}


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------
def run_stage_a(frames, M_borns, FREQS, THETAS0, pool_integers_used):
    with open(LAB99_METRICS_PATH) as f:
        lab99_metrics = json.load(f)
    lab99_classA = {c["frame"]: c["theta"] for c in lab99_metrics["stageB_classA"]}

    theta_provenance = []
    thetas_by_frame = {}
    for seed in FRAME_SEEDS:
        theta = lab99_classA[seed]
        theta_provenance.append({"frame": seed, "theta": theta, "source": "metrics_99.stageB_classA"})
        thetas_by_frame[seed] = np.array(theta, dtype=float)
    A1 = bool(all(seed in lab99_classA for seed in FRAME_SEEDS))
    print(f"A1: theta provenance recorded for frames {list(lab99_classA.keys())}, pass={A1}")

    # A2 - the bridge gate
    channels = collect_pool_channels(K, FREQS, THETAS0, T_RUN)
    ks_stats = []
    for c in range(6):
        stat, _ = kstest(channels[:, c], "uniform")
        ks_stats.append(float(stat))
    max_ks = float(max(ks_stats))
    corrs = []
    for i in range(6):
        for j in range(i + 1, 6):
            r = float(np.corrcoef(channels[:, i], channels[:, j])[0, 1])
            corrs.append(abs(r))
    max_corr = float(max(corrs))
    A2 = bool(max_ks < A2_KS_BAR and max_corr < A2_CORR_BAR)
    print(f"A2: max_ks={max_ks:.5f} (bar {A2_KS_BAR}), max_pairwise_corr={max_corr:.5f} "
          f"(bar {A2_CORR_BAR}), pass={A2}")
    print(f"A2 pool provenance: {len(pool_integers_used)} square-free integers used, "
          f"max={max(pool_integers_used)}")

    # A3 - calibration floors
    markov_floors = {}
    kernel_devs = []
    for idx, seed in enumerate(FRAME_SEEDS):
        cal_stream = lab94.gen_calibration_stream(M_borns[seed], T=T_RUN, seed=CAL_SEED + idx)
        F1_cal, _ = lab91.counted_F(cal_stream, 1)
        D_born_cal = float(lab91.row_max_tv(F1_cal, M_borns[seed]))
        m_floor = float(lab93.markov_score_fn(cal_stream, F1_cal, TAU_MARKOV))
        markov_floors[str(seed)] = m_floor
        kernel_devs.append(D_born_cal)
        print(f"A3 frame {seed}: D_born_cal={D_born_cal:.5f}, markov_floor={m_floor:.5f}")
    kernel_max_dev = float(max(kernel_devs))
    A3 = bool(kernel_max_dev < 0.02)
    print(f"A3 pass: {A3} (kernel_max_dev={kernel_max_dev:.5f})")

    # A4 - round-trip, frame 91's winning theta
    rt_err = roundtrip_test(frames[91], thetas_by_frame[91], K, FREQS, THETAS0, n_steps=1000)
    A4 = bool(rt_err < A4_TOL)
    print(f"A4 roundtrip_max_err={rt_err:.3e}, pass={A4}")

    pass_blocking = bool(A1 and A3 and A4)  # A2 failing routes to POOL-GAP, not VOID
    pass_all = bool(pass_blocking and A2)

    stageA = {
        "theta_provenance": theta_provenance,
        "bridge": {"max_ks": max_ks, "max_pairwise_corr": max_corr, "pass": A2,
                   "pool_integers_used": pool_integers_used,
                   "max_squarefree_used": int(max(pool_integers_used))},
        "calibration": {"markov_floors": markov_floors, "kernel_max_dev": kernel_max_dev, "pass": A3},
        "roundtrip_max_err": rt_err,
        "pass": pass_all,
        "pass_blocking": pass_blocking,
    }
    return stageA, thetas_by_frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building frames, M_born, pool (v2 - square-free + near-resonance screen)...")
    frames = {seed: lab91.haar_frame(seed) for seed in FRAME_SEEDS}
    M_borns = {seed: lab93.compute_M_born(frames[seed]) for seed in FRAME_SEEDS}
    FREQS, pool_integers_used = build_pool_v2()
    THETAS0 = build_thetas0(K)
    print(f"Pool v2 build: {len(pool_integers_used)} integers accepted, max={max(pool_integers_used)}")

    print("\n" + "=" * 60)
    print("STAGE A - Gates")
    print("=" * 60)
    stageA, thetas_by_frame = run_stage_a(frames, M_borns, FREQS, THETAS0, pool_integers_used)
    print(f"\nStage A blocking gates (A1,A3,A4): {'PASS' if stageA['pass_blocking'] else 'FAIL'}")
    print(f"Stage A bridge gate (A2): {'PASS' if stageA['bridge']['pass'] else 'FAIL'}")

    if not stageA["pass_blocking"]:
        metrics = {
            "lab": 100, "stream": "cryptographic-substrate", "kind": "scout-capstone",
            "serves_node": "born_rule_emergence",
            "stageA": stageA, "stageB": [], "stageC": [], "stageD": {"attacks": []},
            "summary": {"J_per_frame": {}, "lab99_targets": {}, "n_frames_under_bar": 0,
                        "n_controls_separating": 0},
            "verdict": "VOID", "verdict_reason": "Stage A blocking gate(s) (A1/A3/A4) failed. Stopping.",
            "leakage_audit": {
                "wrote_outside_lab_folder": False, "theta_from_other_source": False,
                "random_draws_inside_world": False, "emergence_language_used": False,
                "posthoc_threshold_change": False, "metrics_hand_edited": False,
            },
        }
        with open(LAB_FOLDER / "metrics_100_v2.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("\nVerdict: VOID")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("STAGE B - The capstone runs")
    print("=" * 60)
    stageB = []
    main_outcomes = {}
    for seed in FRAME_SEEDS:
        Qf, M_born, theta = frames[seed], M_borns[seed], thetas_by_frame[seed]
        run = run_model_two(Qf, theta, K, FREQS, THETAS0, T=T_RUN, T_transient=T_TRANSIENT)
        main_outcomes[seed] = run["outcomes"]
        cell = compute_cell_metrics(run["outcomes"], M_born)
        cell["frame"] = seed
        cell["M_born"] = M_born.tolist()
        stageB.append(cell)
        print(f"frame {seed}: D_born={cell['D_born']:.5f}, markov={cell['markov_score']:.5f}, "
              f"J={cell['J']:.5f}, degenerate={cell['degenerate']}")
        print(f"  kernel: {np.round(np.array(cell['kernel']), 4).tolist()}")
        print(f"  M_born: {np.round(M_born, 4).tolist()}")

    print("\n" + "=" * 60)
    print("STAGE C - Controls")
    print("=" * 60)
    stageC = []
    for seed in FRAME_SEEDS:
        Qf, M_born, theta = frames[seed], M_borns[seed], thetas_by_frame[seed]

        run_p = run_passive(Qf, K, FREQS, THETAS0, T=T_RUN, T_transient=T_TRANSIENT)
        cell_p = compute_cell_metrics(run_p["outcomes"], M_born)
        sep_p = bool(cell_p["D_born"] > CTRL_PASSIVE_MIN)
        stageC.append({"frame": seed, "control": "C-passive", "D_born": cell_p["D_born"],
                        "markov_score": cell_p["markov_score"], "J": cell_p["J"], "separates": sep_p})
        print(f"frame {seed} C-passive: D_born={cell_p['D_born']:.4f} (bar >{CTRL_PASSIVE_MIN}), separates={sep_p}")

        run_i = run_model_two(Qf, ISO_THETA, K, FREQS, THETAS0, T=T_RUN, T_transient=T_TRANSIENT)
        cell_i = compute_cell_metrics(run_i["outcomes"], M_born)
        sep_i = bool(cell_i["J"] > CTRL_WRONGTHETA_MIN)
        stageC.append({"frame": seed, "control": "C-isotropic", "D_born": cell_i["D_born"],
                        "markov_score": cell_i["markov_score"], "J": cell_i["J"], "separates": sep_i})
        print(f"frame {seed} C-isotropic: J={cell_i['J']:.4f} (bar >{CTRL_WRONGTHETA_MIN}), separates={sep_i}")

        shuf_theta = make_shuf_theta(theta)
        run_s = run_model_two(Qf, shuf_theta, K, FREQS, THETAS0, T=T_RUN, T_transient=T_TRANSIENT)
        cell_s = compute_cell_metrics(run_s["outcomes"], M_born)
        sep_s = bool(cell_s["J"] > CTRL_WRONGTHETA_MIN)
        stageC.append({"frame": seed, "control": "C-shuffled", "D_born": cell_s["D_born"],
                        "markov_score": cell_s["markov_score"], "J": cell_s["J"], "separates": sep_s})
        print(f"frame {seed} C-shuffled: J={cell_s['J']:.4f} (bar >{CTRL_WRONGTHETA_MIN}), separates={sep_s}")

    print("\n" + "=" * 60)
    print("STAGE D - Arrow check")
    print("=" * 60)
    stageD_attacks = []
    for seed in FRAME_SEEDS:
        Qf, M_born, theta = frames[seed], M_borns[seed], thetas_by_frame[seed]
        run = run_model_two(Qf, theta, K, FREQS, THETAS0, T=T_RUN, T_transient=T_TRANSIENT, track_stage_d=True)
        X, y = run["j0_X"], run["j0_y"]
        n = len(X)
        if n < 20:
            stageD_attacks.append({"frame": seed, "r2": None, "n_samples": n})
            print(f"frame {seed}: n_samples={n} (too few)")
            continue
        mid = n // 2
        r2 = lab93.ridge_fit_r2(X[:mid], y[:mid], X[mid:], y[mid:], alpha=1.0)
        stageD_attacks.append({"frame": seed, "r2": r2, "n_samples": n})
        print(f"frame {seed}: n_samples={n}, R2={r2:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    J_per_frame = {str(c["frame"]): c["J"] for c in stageB}
    with open(LAB99_METRICS_PATH) as f:
        lab99_metrics = json.load(f)
    lab99_targets = {str(c["frame"]): c["D_heldout"] for c in lab99_metrics["stageB_classA"]}
    n_frames_under_bar = sum(1 for seed in FRAME_SEEDS if J_per_frame[str(seed)] < J_BAR)
    n_controls_separating = sum(1 for c in stageC if c["separates"])
    summary = {
        "J_per_frame": J_per_frame, "lab99_targets": lab99_targets,
        "n_frames_under_bar": n_frames_under_bar, "n_controls_separating": n_controls_separating,
    }
    print(f"J_per_frame={J_per_frame}, n_frames_under_bar={n_frames_under_bar}/3, "
          f"n_controls_separating={n_controls_separating}/9")

    print("\nGenerating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, seed in zip(axes, FRAME_SEEDS):
        ctrl_cells = {c["control"]: c for c in stageC if c["frame"] == seed}
        labels = ["Main", "C-passive", "C-isotropic", "C-shuffled"]
        vals = [J_per_frame[str(seed)], ctrl_cells["C-passive"]["D_born"],
                ctrl_cells["C-isotropic"]["J"], ctrl_cells["C-shuffled"]["J"]]
        colors = ["tab:green", "tab:gray", "tab:red", "tab:orange"]
        ax.bar(labels, vals, color=colors)
        ax.axhline(J_BAR, color="black", linestyle="--", linewidth=0.9, label=f"J={J_BAR}")
        ax.set_ylabel("J (D_born for C-passive)")
        ax.set_title(f"frame {seed}")
        ax.legend(fontsize=7)
        ax.text(0.02, 0.95, "arc: 0.42 (passive) -> 0.08 (scout mech.) -> "
                             f"{J_per_frame[str(seed)]:.4f} (this lab, v2 pool)",
                transform=ax.transAxes, fontsize=6.5, va="top")
    fig.suptitle("Lab 100 v2 (square-free pool): main-cell J vs three controls per frame; dashed line at J=0.06")
    fig.tight_layout()
    fig.savefig(LAB_FOLDER / "capstone_100_v2.png", dpi=120)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Verdict - first-match order (unchanged ladder, per lab-goal-v2.md)
    # -----------------------------------------------------------------
    A2_pass = stageA["bridge"]["pass"]
    if not A2_pass:
        verdict = "POOL-GAP"
        verdict_reason = (f"A2 bridge gate failed: max_ks={stageA['bridge']['max_ks']:.5f}, "
                           f"max_pairwise_corr={stageA['bridge']['max_pairwise_corr']:.5f}. "
                           f"Pool does not license the MC-to-deterministic transport, regardless of J "
                           f"(J_per_frame={J_per_frame}).")
    elif n_frames_under_bar == 3 and n_controls_separating == 9:
        verdict = "GRADUATE-EXISTENCE"
        verdict_reason = (f"J < 0.06 on 3/3 frames ({J_per_frame}) AND all 9/9 control cells separate "
                           f"per their bars. Existence statement stands at scout level.")
    elif n_frames_under_bar == 2 and n_controls_separating == 9:
        verdict = "PARTIAL-EXISTENCE"
        odd_frame = next(seed for seed in FRAME_SEEDS if J_per_frame[str(seed)] >= J_BAR)
        verdict_reason = (f"J < 0.06 on exactly 2/3 frames with all 9/9 controls separating. "
                           f"Odd frame: {odd_frame} (J={J_per_frame[str(odd_frame)]:.5f}).")
    elif (n_frames_ge_bar := sum(1 for seed in FRAME_SEEDS if J_per_frame[str(seed)] >= J_BAR)) >= 2:
        verdict = "NULL"
        verdict_reason = (f"A2 passed (bridge holds at the marginal-distribution level) but J >= 0.06 on "
                           f"{n_frames_ge_bar}/3 frames: {J_per_frame}. "
                           f"Temporal pool correlations may matter beyond marginals.")
    else:
        verdict = "OTHER"
        verdict_reason = (f"No ladder rule matched exactly: J_per_frame={J_per_frame}, "
                           f"n_frames_under_bar={n_frames_under_bar}/3, "
                           f"n_controls_separating={n_controls_separating}/9.")

    metrics = {
        "lab": 100, "stream": "cryptographic-substrate", "kind": "scout-capstone",
        "serves_node": "born_rule_emergence",
        "stageA": stageA, "stageB": stageB, "stageC": stageC, "stageD": {"attacks": stageD_attacks},
        "summary": summary,
        "verdict": verdict, "verdict_reason": verdict_reason,
        "leakage_audit": {
            "wrote_outside_lab_folder": False, "theta_from_other_source": False,
            "random_draws_inside_world": False, "emergence_language_used": False,
            "posthoc_threshold_change": False, "metrics_hand_edited": False,
        },
    }
    with open(LAB_FOLDER / "metrics_100_v2.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nVerdict: {verdict}")
    print(f"Reason: {verdict_reason}")
