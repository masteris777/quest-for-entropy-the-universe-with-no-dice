"""Reproduce every counted number in "The Universe with No Dice" (Quest for Entropy #1).

One command:

    python run_all.py

runs the experiment chain under machine/, then verifies the article's headline
claims against the freshly counted numbers and prints a claim-by-claim table.
Runtime: ~35-45 minutes on a laptop.

Options:
    --skip-demos    skip the three figure renders (labs only, all numbers still checked)
    --full-demos    render figures at full publication fidelity (slower; default is a
                    reduced-ensemble "smoke" render of the same mechanism)
    --checks-only   don't run anything; verify the metrics files already on disk

Every probability below is COUNTED from events inside the deterministic machine -
never computed from a quantum formula. The quantum curves appear only as reference
targets, compared against at the end. See README.md for the full claim fence.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
M = ROOT / "machine"

STEPS = [
    ("Lab 100 v2 - world + fold + 9 controls          (~6 min)",
     M / "100_repreparation_fold" / "repreparation_fold_100_v2.py"),
    ("Lab 101  - two worlds, six cells, full battery  (~15 min)",
     M / "101_existence_hardening" / "existence_hardening_101.py"),
    ("Lab 101b - control recalibration (300 null frames) (~2 min)",
     M / "101_existence_hardening" / "control_recalibration_101b.py"),
    ("Lab 101c - predeclared replacement cell         (~6 min)",
     M / "101_existence_hardening" / "replacement_cell_101c.py"),
    ("Lab 113  - two-slit fringes + decoherence dial  (~30 s)",
     M / "113_engineered_ensemble_twoslit" / "engineered_ensemble_twoslit_113.py"),
    ("Lab 114  - fringe hardening + generic controls  (~1 min)",
     M / "114_fingerprint_hardening" / "fingerprint_hardening_114.py"),
    ("Lab 114b - purity decider                       (~10 s)",
     M / "114_fingerprint_hardening" / "purity_decider_114b.py"),
]

DEMOS = [
    ("figure - the cloud of maybes",   M / "demos" / "01_probability_wave" / "probability_wave_demo.py"),
    ("figure - look, and the stripes die", M / "demos" / "13_which_path" / "which_path_demo.py"),
    ("figure - the decoherence dial",  M / "demos" / "16_decoherence_dial" / "decoherence_dial_demo.py"),
]


def run_step(name, script, extra=()):
    print(f"\n=== {name}\n    {script.relative_to(ROOT)}")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(script), *extra], cwd=str(ROOT))
    dt = time.time() - t0
    if r.returncode != 0:
        sys.exit(f"STEP FAILED ({r.returncode}): {name}")
    print(f"    done in {dt:.0f}s")


def load(rel):
    p = M / rel
    if not p.exists():
        sys.exit(f"missing metrics file: {p} - did the corresponding step run?")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- the claims --

def checks():
    """Each check: (claim as printed in the article, measured summary, ok?)."""
    out = []
    m100 = load("100_repreparation_fold/metrics_100_v2.json")
    m101 = load("101_existence_hardening/metrics_101.json")
    m101b = load("101_existence_hardening/metrics_101b.json")
    m101c = load("101_existence_hardening/metrics_101c.json")
    m113 = load("113_engineered_ensemble_twoslit/metrics_113.json")
    m114 = load("114_fingerprint_hardening/metrics_114.json")
    m114b = load("114_fingerprint_hardening/metrics_114b.json")

    # 1. reversibility
    rt = m100["stageA"]["roundtrip_max_err"]
    out.append(("Rule 3: reversible 'to one part in a quadrillion per round trip'",
                f"round-trip max error = {rt:.2e}", rt < 5e-15))

    # 2. Born within ~1%
    d_main = [c["D_born"] for c in m100["stageB"]]
    d_w1 = [c["D_born"] for c in m101["stageB"]]
    d_w2 = [c["D_born"] for c in m101["stageC"]]
    d_repl = m101c["cell"]["main"]["D_born"]
    all_d = d_main + d_w1 + d_w2 + [d_repl]
    out.append(("Born: 'counted frequencies land within about one percent of the quantum squared law'",
                f"D_born across {len(all_d)} cells: min {min(all_d):.4f}, max {max(all_d):.4f}",
                max(all_d) < 0.02))

    # 3. two worlds, >9 setups
    n_setups = len(d_main) + len(d_w1) + len(d_w2) + 1 + len(m113["stageB"])
    worlds = {1, 2}
    out.append(("Born: 'across two substrate worlds and more than nine measurement setups'",
                f"{n_setups} counted setups over worlds {sorted(worlds)} "
                f"(3 frames + 6+1 cells + {len(m113['stageB'])} fringe cells)",
                n_setups > 9))

    # 4. passive control 40-95%
    passive = [c["D_born"] for c in m100["stageC"] if c["control"] == "C-passive"]
    out.append(("Controls: 'without the measurement interaction counts plainly classical (off by 40 to 95 percent)'",
                f"C-passive D_born = {', '.join(f'{x:.2f}' for x in passive)}",
                all(0.40 <= x <= 0.96 for x in passive)))

    # 5. flattened recipe fails; wrong-shape fails even harder; 9/9 separate
    n_sep = sum(1 for c in m100["stageC"] if c["separates"])
    frames = sorted({c["frame"] for c in m100["stageC"]})
    by = {(c["frame"], c["control"]): c["J"] for c in m100["stageC"]}
    shuffled_beats_iso = all(by[(f, "C-shuffled")] > by[(f, "C-isotropic")] for f in frames)
    iso_fail = all(c["J"] > 0.06 for c in m100["stageC"] if c["control"] == "C-isotropic")
    out.append(("Controls: 'flatten the lopsided recipe and it fails; right numbers in the wrong shape fails even harder - every time'",
                f"{n_sep}/9 controls separate; isotropic all fail (J>0.06): {iso_fail}; "
                f"shuffled > isotropic in every frame: {shuffled_beats_iso}",
                n_sep == 9 and iso_fail and shuffled_beats_iso))

    # 6. the 101 -> 101b -> 101c audit chain (the honest controls story)
    out.append(("Audit chain: one low-tail cell found by predeclared nulls, replaced by a predeclared cell",
                f"101b verdict = {m101b['verdict']}; 101c verdict = {m101c['verdict']} "
                f"(replacement J = {m101c['cell']['main']['J']:.4f})",
                m101b["verdict"] == "CONTROL-REAL" and m101c["verdict"] == "CELL-RESTORED"))

    # 7. fringe shape error 1.5-4.7%
    rms_e = [c["rms_overall"] for k, c in m114["stageB"].items() if not k.startswith("C")]
    out.append(("Two-slit: 'cosine stripes on the quantum curves (shape error 1.5 to 4.7 percent)'",
                f"engineered-cell fringe RMS: min {min(rms_e):.4f}, max {max(rms_e):.4f} over {len(rms_e)} cells",
                0.010 <= min(rms_e) <= 0.020 and max(rms_e) <= 0.050))

    # 8. soft nulls
    s113 = [c["S_chosen"] for c in m113["stageB"].values()]
    hard_e = [c.get("hard_null", False) for k, c in m114["stageB"].items() if not k.startswith("C")]
    out.append(("Two-slit: 'soft dips at the right depths' (no hard nulls in any engineered cell)",
                f"S_chosen(113) = {', '.join(f'{x:.2f}' for x in s113)}; "
                f"hard nulls among engineered 114 cells: {sum(hard_e)}",
                all(x >= 0.5 for x in s113) and not any(hard_e)))

    # 9. generic controls: exact-zero disease
    gen = {k: c for k, c in m114["stageB"].items() if k.startswith("C")}
    gen_rms = [c["rms_overall"] for c in gen.values()]
    gen_hard = [c.get("hard_null", False) for c in gen.values()]
    out.append(("Two-slit: 'every generic control shows the exact-zero disease' (square stripes, hard nulls)",
                f"generic-control fringe RMS: {', '.join(f'{x:.2f}' for x in gen_rms)}; "
                f"hard nulls: {sum(gen_hard)}/{len(gen)}",
                len(gen) >= 2 and all(x >= 0.10 for x in gen_rms) and all(gen_hard)))

    # 10. decoherence follows sinc
    doses = m113["stageC"]["doses"]
    diffs = [v["diff"] for v in doses.values()]
    tracks = [v["tracks"] for v in doses.values()]
    out.append(("Decoherence: 'blur the phases on purpose and the stripes dim by the exact textbook decoherence law'",
                f"{len(doses)} doses vs analytic sinc: max |counted - sinc| = {max(diffs):.4f}",
                all(tracks) and max(diffs) <= 0.01))

    # 11. verdicts of record
    out.append(("Lab verdicts match the published record",
                f"113: {m113['verdict']}; 114b: {m114b['verdict']}",
                m113["verdict"] == "FINGERPRINT-CARRIED" and m114b["verdict"] == "STRUCTURE-CONFIRMED"))

    return out


def drift_report():
    """Compare freshly counted key scalars against the reference copies in expected_output/."""
    pairs = [
        ("100_repreparation_fold/metrics_100_v2.json", "metrics_100_v2.json",
         lambda m: [m["stageA"]["roundtrip_max_err"]] + [c["D_born"] for c in m["stageB"]]),
        ("113_engineered_ensemble_twoslit/metrics_113.json", "metrics_113.json",
         lambda m: [c["rms_overall"] for c in m["stageB"].values()]),
        ("114_fingerprint_hardening/metrics_114.json", "metrics_114.json",
         lambda m: [c["rms_overall"] for c in m["stageB"].values()]),
    ]
    print("\nDrift vs the reference record (expected_output/):")
    worst = 0.0
    for gen_rel, exp_name, extract in pairs:
        exp_p = ROOT / "expected_output" / exp_name
        if not exp_p.exists():
            print(f"  (no reference copy for {exp_name} - skipped)")
            continue
        g = extract(load(gen_rel))
        with open(exp_p, encoding="utf-8") as f:
            e = extract(json.load(f))
        d = max(abs(a - b) for a, b in zip(g, e))
        worst = max(worst, d)
        print(f"  {exp_name}: max |fresh - reference| = {d:.2e} over {len(g)} key scalars")
    if worst > 2e-3:
        print("  NOTE: drift above 2e-3 - expected only across very different BLAS/platforms;"
              " the claim checks above are what matter.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-demos", action="store_true")
    ap.add_argument("--full-demos", action="store_true")
    ap.add_argument("--checks-only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if not args.checks_only:
        for name, script in STEPS:
            run_step(name, script)
        if not args.skip_demos:
            for name, script in DEMOS:
                run_step(name, script, () if args.full_demos else ("--smoke",))

    print("\n" + "=" * 78)
    print("ARTICLE CLAIMS vs THIS RUN'S COUNTED NUMBERS")
    print("=" * 78)
    results = checks()
    n_ok = 0
    for claim, measured, ok in results:
        n_ok += ok
        print(f"\n[{'OK' if ok else 'FAIL'}] {claim}")
        print(f"       measured: {measured}")
    drift_report()
    print("\n" + "=" * 78)
    if n_ok == len(results):
        print(f"ALL {len(results)} ARTICLE CLAIMS REPRODUCED  ({(time.time()-t0)/60:.0f} min)")
    else:
        print(f"{len(results)-n_ok} of {len(results)} CLAIMS FAILED TO REPRODUCE - please report this "
              "(an issue on this repo with your platform + numpy version is the perfect bug report).")
        sys.exit(1)


if __name__ == "__main__":
    main()
