# The Universe with No Dice — companion repository

**Article:** [Quest for Entropy #1 — "The Universe with No Dice"](https://questforentropy.com/p/the-universe-with-no-dice) · also on [Substack](https://questforentropy.substack.com/p/the-universe-with-no-dice)

**Series:** [#2 The Machine](https://github.com/masteris777/quest-for-entropy-the-machine) →

Evidence repo for **Quest for Entropy #1: [“The Universe with No Dice”](article/article.md)** —
a fully deterministic, classical model universe whose *counted* statistics land within
about one percent of quantum mechanics’ squared law, draw soft-nulled two-slit stripes,
and dim by the textbook decoherence law — with no random number drawn anywhere and no
probability ever computed from a quantum formula.

## Run it

```
pip install -r requirements.txt
python run_all.py
```

~35–45 minutes on a laptop. Runs the full experiment chain, then checks **every number
in the article** against the freshly counted results and prints a claim-by-claim table:
the Born distances (~1%), the two-substrate-world battery, the three failing controls
(including the “off by 40 to 95 percent” passive watcher), the fringe shape errors
(1.5–4.7%), the soft nulls, the generic controls’ exact-zero disease, and the sinc
decoherence law.

`python run_all.py --checks-only` re-verifies without re-running; `--skip-demos` skips
figure renders; `--full-demos` renders figures at full fidelity.

## What is in here

```
machine/          the experiment scripts and their frozen inputs (metrics_92/94/99.json —
                  constants and the optimizer-found ensemble parameters, fixed before
                  any fringe was ever computed)
expected_output/  the reference metrics + figures the article quotes — diff before
                  running anything
article/          the article and its figures
run_all.py        the runner + the claim checker
MANIFEST.sha256   checksums of the files above
```

The chain, in run order: Lab 100v2 (the world, the fold, 9 controls) → Lab 101 (six
cells across two substrate worlds) → Labs 101b/101c (a control audit that found one
low-tail cell and replaced it by predeclared criteria — kept in full) → Lab 113
(two-slit fringes + the decoherence dial) → Labs 114/114b (fringe hardening + the
generic-ensemble controls).

## Scope

A demonstration, not a discovery about nature. The full scope fence (“What this does
NOT claim”) and how this was made — including the AI-assisted workflow — are in
[the article](article/article.md).

## License

Code: MIT. Article text and figures: CC BY 4.0.
