# Overnight Training — v5 Engine (2026-07-18)

Every v4-and-earlier record, surrogate, and paper number is stale under the
v5 physics (Watson eye MTF, walk-off chromatics, Fresnel de-dup,
re-interaction de-rating, pinned n — `docs/PHYSICS_VALIDATION.md` §5).
This is the checklist that regenerates everything overnight and tells you
what to expect in the morning.

## 1. Pre-flight (5 minutes, do these before bed)

From the project root:

```bash
pip3 install torch numpy matplotlib grcwa autograd   # if not already present

python3 physics/waveguide_physics.py   # must print ENGINE v5 banner, finite
                                       # gradients, Watson 0.4884, unguided
                                       # T ~1e-21, and the RCWA record check
python3 physics/validate.py            # must print 11/11 tests passed
```

If either fails, stop — do not train on a broken engine. The RCWA grid
(`physics/rcwa_eta_grid.npz`) ships in the repo; only rebuild it
(`python3 physics/calibrate_rcwa.py`, ~hours) if the file is missing or you
changed the grid axes.

Notes that make the night smooth:

* **Stale checkpoints are handled automatically.** Every checkpoint stores a
  64-design physics probe; the first v5 surrogate seed replaces any
  checkpoint trained under older physics, and every loader refuses stale
  weights with an explicit retrain message. You do not need to delete
  anything by hand.
* **Records cannot mix.** All output tables/records are keyed on
  `ENGINE_VERSION` (`best_design_ever_v5.csv`, `surrogate_runs_v5.csv`,
  `training_runs_v5.csv`, `design_rcwa_check*_v5.csv`). v3/v4 files stay as
  archives and are never compared against v5 numbers.
* Disk/RAM: 150k-sample datasets are ~10 MB; any laptop is fine. Keep the
  machine on AC power (`caffeinate` is used automatically on macOS).

## 2. Launch (one command)

```bash
bash overnight.sh                # >= 12 hours (default)
HOURS=16 bash overnight.sh       # longer night
HOURS=0  bash overnight.sh       # core 5-seed protocol only, no bonus laps
```

Watch from another terminal: `tail -f overnight_log.txt`

What it runs (in order):

| Stage | What | Why it's in the paper |
|---|---|---|
| 1 | forward surrogate × 5 seeds (150k samples, 250 epochs) | surrogate R² table |
| 2 | tandem inverse through the surrogate × 5 seeds | headline method |
| 3 | tandem inverse through exact physics × 5 seeds | ablation arm |
| 4 | neural-adjoint (4000 starts × 600 steps) + gradient + Pareto baselines | record + baseline table |
| 5 | figures, 3D model, memorization audit, HTML report | figures |
| 6 | bonus laps until the time budget is spent | extra seeds/statistical power |

## 3. Morning checklist (what "good" looks like under v5)

Open `results_report.html`. Sanity anchors (from the v5 gradient probe —
these bound what any search can honestly report):

* **MTF@40cyc/mm record ≈ 0.44–0.47** (ceiling ≈ 0.466; the Watson eye term
  alone caps the product at 0.4884 for a 3 mm pupil). If you see anything
  near the old 0.78, something loaded a stale checkpoint — check the log
  for the probe-refusal message.
* **Throughput FOM record ≈ 1.0–1.4%** (ceiling ≈ 1.35%). This is a
  RELATIVE figure of merit (single eyebox position, no exit-pupil
  expansion) — the paper frames it against the ~10% @20° / ~3% @30° system
  efficiencies of *Light Sci. Appl.* 13 (2024), it is not a device
  efficiency claim.
* **Chromatic pupil walk-off ≈ 1.5–2.5 mm** (floor ≈ 1.52 mm). Units are mm
  at the eye pupil, not degrees — the old 28.8° metric is retired.
* Surrogate R² per metric should recover ≥ 0.99 at 150k/250 (the quick
  smoke run's low R² is expected — 15 epochs on 8k samples).
* `results/best_design_ever_v5.csv` should show n = 1.49369 exactly (pinned
  material constant) and an interior depth optimum (~180–210 nm, not the
  400 nm bound).

Then re-verify the winners rigorously (minutes, not hours):

```bash
python3 physics/rigorous_solver.py --designs-na   # RCWA check of NA winners
python3 physics/rigorous_solver.py --designs      # RCWA check of baseline winners
```

## 4. After the run — paper resync (do NOT skip)

1. Replace every number in `docs/ABSTRACT.md` and `paper/waveguide_paper.tex`
   flagged by the v5 notice: MTF headline, throughput FOM + framing,
   walk-off metric (mm), surrogate R², tandem MSEs, record design row.
2. The paper's MTF-cascade section must describe the v5 cascade: Watson
   (2013) mean-eye term at the walk-off-apodized pupil; chromatic term =
   period-mismatch residual only; no flat-interface Fresnel² in the coupled
   path; re-interaction survival term with the Zhao et al. (2024) citation.
   The long v3-era discussion of the 0.543 chromatic floor / fringe lottery
   describes retired physics — cut or rewrite it as a lessons-learned note.
3. Keep the disclosure that an independent review caught the Watson
   mislabeling; reviewers reward that, and the tex already contains the
   corrected description.
4. Do not quote MTF as a rigorous number — the roughness/grating/coupler
   MTF coefficients and the pupil-overlap chord model remain flagged `# SYNC`
   L1 heuristics (review §2.9). Quote it as a model-consistent design
   metric whose *rankings* are physics-anchored.

## 5. Known limitations that stay open overnight (documented, not hidden)

* Exit-pupil expansion / eyebox uniformity is not modeled; the throughput
  FOM is a ranking proxy (review §2.2).
* The RCWA grid is normal-incidence and reused for the out-coupler by
  reciprocity (review §2.8) — a dedicated substrate-side solve is the
  cheapest remaining L2 gap.
* σ ∈ [0.7, 1.1] nm assumes spin-coated PMMA (Nilsen et al. 2025);
  untreated PMMA is rougher and out of scope (review §2.5).
* The interactive `waveguide_designs_3d.html` embeds the v2-era scalar JS
  engine for illustration only; its live numbers are not v5.
