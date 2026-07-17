# Next Steps — after the v3 (RCWA-calibrated) physics revision

**Written:** 2026-07-17, the day the v3 engine landed.
**Read first:** `README.md` ("Critical next steps"), `docs/research_framework.md`
(§6 experiment plan), `docs/CITATIONS.md` (Groups G–H).

---

## 1. What just changed (v3, 2026-07-17)

The engine's grating-coupling term is no longer scalar theory. In PMMA mode,
`waveguide_physics.py` now interpolates the first-order coupling efficiency
η₁(n, period, depth, duty; λ, polarization) from a rigorous `grcwa` grid of
90,090 vectorial solves (`physics/rcwa_eta_grid.npz`, built by
`physics/calibrate_rcwa.py`). The grid carries its own refractive-index axis
because η₁(TM) was measured to move ~20% across the PMMA bounds
n ∈ [1.48, 1.50]. Off-grid interpolation error is audited in
`results/rcwa_calibration_check.csv`. Consequences to keep in mind:

- **Every v2-trained checkpoint is stale.** The physics-probe system refuses
  them automatically (`SystemExit` on load) — this is correct behavior, not a
  bug. Retraining under v3 is the fix (see §2).
- **v2 and v3 numbers must never be pooled.** Records, run tables, and RCWA
  check files are keyed on `ENGINE_VERSION` (`best_design_ever_v3.csv`,
  `surrogate_runs_v3.csv`, `training_runs_v3.csv`, `design_rcwa_check_na_v3.csv`).
  The v2-era files stay in `results/` untouched, as the audit trail behind the
  "why v3 was necessary" paragraph of the paper.
- **Absolute transmission dropped ~25×** (η enters twice: in-coupler ×
  out-coupler). That is physics honesty, not regression: the scalar engine was
  overstating coupling ~5× per grating at 532 nm. The correct external
  comparison window is the ~1–10% system-efficiency range reported for
  conventional diffractive combiners (CITATIONS Group G4, item 35).
- **The depth story flipped** — the optimizer now finds an interior depth
  optimum (rigorous η peaks near ~200 nm) instead of pinning depth at the
  400 nm bound the way scalar theory forced it to. This is a headline result:
  the same inverse pipeline, fed rigorous physics, chooses a *different and
  correct* design.

## 2. How to continue training

Already done on 2026-07-17, same day as the v3 revision (do NOT redo):
surrogate seeds 0–4 at the full 150k/250 protocol
(`results/surrogate_runs_v3.csv`), tandem seed 0 in BOTH decoder arms at
150k/400 (`results/training_runs_v3.csv`), the v3 memorization audit, the
neural-adjoint record search, and the rigorous verification of its winners.

What training remains: inverse-net seeds 1–4 in both decoder arms, to power
the paper's 5-v-5 tandem comparison table the way the v2 tables were powered.
Either run the targeted loop:

```bash
for S in 1 2 3 4; do
  python3 networks/train_inverse.py --pmma --decoder surrogate \
      --samples 150000 --epochs 400 --batch 256 --seed $S
  python3 networks/train_inverse.py --pmma --decoder physics \
      --samples 150000 --epochs 400 --batch 256 --seed $S
done
```

or just run the full overnight protocol, which includes it (extra surrogate
seeds land as bonus statistical power; records only ever improve):

```bash
bash overnight.sh                 # >= 12 h: 5 surrogate seeds + 5+5 tandem
HOURS=16 bash overnight.sh        # longer night = more bonus seeds/records
HOURS=0  bash overnight.sh        # core protocol only (~fastest honest run)
```

Notes specific to the v3 transition:

1. The first Stage-1 seed will automatically **replace** the stale v2
   `forward_surrogate.pt` (probe mismatch ⇒ auto-replace on retrain,
   regardless of val score). Nothing to clean up by hand.
2. New rows land in `surrogate_runs_v3.csv` / `training_runs_v3.csv`. The
   paper's Test-D stability panel and the 5-v-5 decoder comparison need the
   standard protocol config (150000 samples / 250 epochs surrogate,
   150000/400 inverse) — `overnight.sh` already uses exactly that.
3. `networks/audit_memorization.py` now reads the v3 run table and will
   refuse to run until protocol rows exist — run it after the first full
   overnight, then it auto-refreshes `figures/memorization_audit.png`.
4. Trim any `--quick` smoke rows out of the v3 run CSVs before using them as
   protocol tables (same hygiene rule as before; quick rows are identifiable
   by `samples=8000/epochs=15`).
5. Do **not** rebuild `physics/rcwa_eta_grid.npz` unless the coupler geometry
   model changes (profile, incidence, material). If it is rebuilt, the probe
   system will invalidate checkpoints again — budget a retrain in the same
   session. `--quick` grids are for smoke tests only and must not be committed.
6. Seed count: 5 seeds per arm satisfies the protocol. The 73-seed v2
   experience says the surrogate saturates immediately; spend bonus laps on
   record hunting, not seed inflation.

## 3. Remaining paper work, in priority order

1. **Reverse-engineering case study** (framework §6.6) — HEADLINE DEMO, still
   the single highest-leverage missing piece. Take published commercial-class
   waveguide specs (Kress & Chatterjee 2020/2021 review) as target y*, recover
   θ̂ with the v3-trained inverse net, compare against known material choices.
2. **Out-of-distribution test** (§6.4) — request specs better than anything in
   the training distribution; report honestly whether the network extrapolates
   or saturates at design bounds. With v3's interior depth optimum, bound
   saturation is now informative rather than an artifact.
3. **Direct θ-regression baseline table** (§6.3a) — the naive baseline is
   already trained inside `train_inverse.py` every run; promote it into a
   proper table (per-metric MAE next to the tandem) for the non-uniqueness
   argument.
4. **Equal-wall-time baseline framing** (§6.3b) — confirm the
   `optimize_pmma.py` comparison is quoted at matched compute budgets.
5. **Physics unit tests vs Paper 1** (§7) — every `# SYNC` constant must
   reproduce Paper 1's published numbers (Si₃N₄ loss 93.37–93.39%, PMMA MTF
   0.6426–0.6430). Publication-blocking gate. Note: v3 changed the coupling
   term, so reconcile the *remaining* SYNC constants (roughness-MTF, S(Lc),
   ACCEPT_SIN, RESID_DISP, coupler-MTF coefficients), not η₁.
6. **Design-manifold analysis** (§6.5) — see §4c below for the upgraded plan.
7. Repo hygiene: ~60 MB of bonus checkpoints in `checkpoints/` — trim or
   gitignore before linking the repo as supplementary material.

## 4. Research menu for the two allowed follow-on changes

The rule for this phase: at most 1–2 more substantive changes before the
manuscript freeze, both in service of physics accuracy / analysis depth.

### (a) Physics accuracy — DONE this revision (v3)
The RCWA-calibrated coupling term was the highest-impact accuracy upgrade on
the table (README item 2, now resolved). What it deliberately leaves out, and
what a **v4** would look like if reviewers demand it:

- **Differentiate through the rigorous solver itself** instead of through the
  interpolation table: TorchRDIT (CITATIONS #42, ~16× faster than
  eigendecomposition RCWA, exact gradients), torcwa (#43), or meent (#44).
  This removes interpolation error entirely and extends naturally to slanted /
  blazed / freeform profiles. Cost: heavier dependency, GPU-preferred, and
  every training label becomes a rigorous solve — mitigate with the
  physics-based transfer learning recipe already shortlisted (#41): pretrain
  on the v3 engine, fine-tune on few rigorous labels.
- **n-dependence of η₁**: already handled — the grid carries a 3-point n
  axis after measurement showed η₁(TM) moves ~20% across the PMMA index
  bounds (a linear-in-n effect the interpolation captures well).
- **Conical / slanted geometry**: `rigorous_solver.py` already has slanted and
  blazed profile builders; topology optimization of grating profiles under
  conical incidence (#45) is the systematic version. This is follow-up-paper
  scope, not this paper.
- **Engine hot-path micro-optimizations** (only if training scale grows):
  `forward_model` interpolates η five times per call where two suffice (the
  coupling term does not depend on field angle, and unpol = (TE+TM)/2), and
  the 16-corner interpolation loop could be one batched gather. Both are
  pure-speed refactors with zero physics effect; current training is ~9 min
  per protocol seed, so they were deliberately left simple.

### (b) Optimization accuracy — boundary-loss neural adjoint
The v2 record pinned n, α, L_corr (and, artifactually, depth) at bounds. The
boundary-loss NA variant (AEM_DIM_Bench, CITATIONS #30) penalizes wall-hugging
so the search reports honest interior optima where they exist. Small change
(one penalty term in `neural_adjoint.py`), directly answers a predictable
reviewer question. Good candidate for the second allowed change if wanted.

### (c) Topological analysis of the design manifold (§6.5, upgraded plan)
The one-to-many y→θ structure is the paper's motivating problem; analyze it
with actual topology rather than scatter plots:

- Sample the inverse solution family for fixed targets y* (input-noise
  sampling through the tandem net, or multi-start NA finalists filtered to
  equal spec), then run **persistent homology** on the solution point cloud
  (ripser / giotto-tda). H₀ persistence = how many *disconnected* design
  families produce the same spec (the honest multiplicity count); H₁ =
  loop/ridge structure (continuous trade-off directions, e.g. the
  depth-duty ridge along constant η₁). Precedent for TDA on photonic design
  spaces: persistent homology for photonic band structures, APL Photonics 6,
  030802 (CITATIONS #47).
- Cross-check the count against the `geometry diversity` basin indicator
  already printed by `optimize_pmma.py`.
- Deliverable: one figure (persistence diagram + annotated design families)
  and one paragraph; it converts §6.5 from "nice-to-have visualization" into
  a quantitative claim about the inverse problem's structure.

## 5. Venues & timeline (unchanged)

Optics Express first (Nov–Dec 2026), Applied Optics / Optics Continuum
fallbacks, SPIE AR|VR|MR (Photonics West, January) as the conference track,
Regeneron STS / ISEF in parallel. arXiv preprint at submission time. The
current draft abstract lives in `docs/ABSTRACT.md`; refresh its numbers after
the first full v3 overnight before circulating it anywhere.
