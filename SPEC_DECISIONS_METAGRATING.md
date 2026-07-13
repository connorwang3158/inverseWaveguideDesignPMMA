# Spec Addendum — Locked Decisions & Binding Amendments
**Applies to:** "Design Spec — Inverse Design of a PMMA Metagrating In-Coupler
(Topology-Optimized vs. SRG Baseline)", 2026-07-12
**Status:** Spec review COMPLETE. The four open decisions below were resolved
by the author on 2026-07-12; the amendments are binding for implementation
planning. Supersedes the corresponding items in `SPEC_REVIEW.md`.

---

## Locked decisions

**D1 — Freeform design space: ρ(x,z) multilayer.**
The frontier arm's density field spans multiple sublayers within the shared
depth budget h, with a **no-overhang (monotone-profile) constraint** for
nanoimprint mold release. Consequence: the freeform design space strictly
contains both binary and slanted-SRG geometries, so freeform-optimum ≥
SRG-optimum is guaranteed up to optimizer convergence, and the SRG-vs-
metagrating gap is a rigorous statement, not a horse race. (§5.3, §5.5)

**D2 — Polarization: TE is the pre-registered primary objective.**
Optimize TE; report TM and unpolarized alongside for every design. Rationale:
polarized LED/LCoS projector sources; repo RCWA data shows TE/TM optima are
geometrically far apart (binary PMMA, Λ=438 nm: TE peak ≈200 nm depth, TM
≈450 nm, ~4× splitting), so the choice must precede results. (§5.3, §5.4)

**D3 — Period: matched bounds, free per arm.**
Both arms receive identical bounds **Λ ∈ [370, 500] nm** (inside the 532 nm
guided window λ/n < Λ < λ → 357–532 nm) and each arm reaches its own optimum.
Matched constraints, not a matched pinned value; report each optimum's guided
angle θ_d. (§5.3)

**D4 — Robust formulation: mandatory.**
The topology arm's headline result is the **robust (eroded/nominal/dilated)
optimum**; the nominal optimum is reported alongside. The nominal-vs-robust
gap is itself a reportable finding on low-index design fragility. (§5.5, §5.7)

## Binding amendments carried over from review

**A1 — Fabricability (§5.5):** minimum-feature-size filtering + Wang-Lazarov-
Sigmund projection AND the D1 no-overhang constraint. No floating islands or
enclosed air pockets in accepted designs (connectivity check on the final
binary pattern).

**A2 — Performance-bound bracketing (§5.8):** report the topology optimum as
the *lower* bound and the Zhao et al. 2024 reciprocity/energy limit
(Opt. Express 32, 12340–12357; DOI 10.1364/OE.519027) as the *upper* bound
for the matched geometry. Add the in-coupler re-interaction caveat to §7
(single-pass η ≠ system in-coupling).

**A3 — Solver rigor (§6):**
- Validate autodiff gradients against finite differences at random density
  fields before any optimization run.
- Headline numbers come from re-simulating the fully binarized (β→∞) design,
  never the relaxed grayscale field.
- Run the Fourier-order convergence study on a freeform pattern at w_min
  (repo data: binary PMMA converges by nG=41; w_min-scale features will
  require substantially more orders).
- Energy conservation |R+T−1| < 1e−6 checked at every evaluation (repo
  baseline: grcwa achieves ~1e−14).

**A4 — Pre-committed outcomes (§2):** "topology optimization yields no
significant gain over the optimized SRG on low-index PMMA" is a valid,
publishable outcome, stated in the spec before any runs.

## Assumptions confirmed against this repo (§12)

532 nm single-λ ✓ · air-clad pure-PMMA relief matches `rigorous_solver.py`
geometry ✓ · 1D-periodic acceptable for paper one ✓ · out-coupler symmetry
deferred but flagged (out-coupling occurs at the guided angle) ✓ · w_min and
h from cited nanoimprint literature (to be sourced during implementation
planning) ✓

## Repo assets the implementation inherits

- `rigorous_solver.py`: grcwa cross-check engine, slanted staircase profile,
  convergence/energy machinery (9/9 tests in `validate.py`).
- `waveguide_physics.py` v2: guided-window/TIR mathematics, `fov_window_deg`
  for the index-ceiling figure, system-level L1 envelope.
- `rcwa_depth_sweep.csv`, `design_rcwa_check.csv`: binary-grating anchors the
  SRG arm must beat.
- Queued (author request, separate work item): live self-updating training
  dashboard — HTML output, per-metric hall-of-fame records (MTF /
  transmission / chromatic), real-time model updates.
