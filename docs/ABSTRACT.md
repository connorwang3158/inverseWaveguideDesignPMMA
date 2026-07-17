# Abstract (draft v1 — 2026-07-17, under the v3 RCWA-calibrated engine)

**Status / where this can go right now**
- **SPIE Photonics West AR|VR|MR 2027** (San Francisco, Jan 30–Feb 4 2027):
  abstract deadline **22 July 2026**. The current results are sufficient for
  this submission — SPIE wants ~250 words + a short supporting summary, not
  final tables. Submit the version below.
- **Optics Express** (target Nov–Dec 2026): this same abstract works as the
  manuscript draft abstract, but the paper itself still needs the
  reverse-engineering case study, the OOD test, the baseline table, and the
  Paper-1 `# SYNC` unit tests (see `docs/NEXT_STEPS.md` §3) before submission.
- All numbers below are from the full 5-seed v3 protocol run of 2026-07-17
  (surrogate seeds 0–4 at 150k/250; tandem both decoder arms at 150k/400;
  neural-adjoint 4000×600; record RCWA-verified per polarization).

---

## Title

Physics-anchored tandem neural networks for multi-objective inverse design
of low-cost polymer diffractive AR waveguides

## Abstract (~240 words, journal/conference)

Diffractive waveguide combiners dominate consumer AR displays, yet their
design couples material parameters and grating geometry to system-level image
quality in ways that make manual iteration slow, and existing neural
inverse-design approaches invert a single component-level metric through a
learned forward surrogate that itself carries approximation error. We present
a neural inverse-design framework for PMMA surface-relief-grating waveguides
that inverts a joint, perceptually grounded system specification — MTF at
40 cyc/mm, double-coupler transmission, lateral chromatic spread, and
transmission at the field edge — and anchors every stage to exact physics. A
differentiable forward model enforces the total-internal-reflection guiding
condition, resolves Fresnel and grating responses by polarization, and
replaces scalar grating-coupling theory — which we show overestimates
first-order coupling efficiency 5–15× at the TIR-mandated sub-wavelength
periods (430–449 nm) — with a polarization-resolved efficiency term
calibrated to 90,090 rigorous RCWA solutions over index, period, depth, and
duty cycle (mean interpolation error 5.6×10⁻⁴, audited against off-grid
solves). A forward surrogate network reproduces this engine with held-out
R² ≥ 0.9989 on every metric across five independent retrainings and passes a
four-test memorization audit (never-seen error 1.2× training error); a
tandem inverse network trained through the frozen forward model resolves the
one-to-many inverse problem, recovering designs that meet requested
specifications in milliseconds with 0.8–7% median per-metric error — where a
direct design-regression baseline fails at 12–47%, the signature of inverse
non-uniqueness — and a neural-adjoint search over the trained surrogate
discovers record designs that an independent exact-physics gradient search
reproduces. Rigorous calibration qualitatively
changes the optimum: the best design's grating depth moves from the
scalar-theory bound (400 nm) to an interior optimum at ≈200 nm and its duty
cycle from 0.5 to ≈0.41, with transmission quoted at rigorous rather than
scalar accuracy (1.3% double-coupler, in line with reported single-digit
system efficiencies for diffractive combiners) and MTF at 40 cyc/mm
improving to 0.78. The framework
provides a template for physics-honest, system-level inverse design of
consumer-grade polymer waveguides.

## One-paragraph significance statement (for SPIE's "summary" box)

Prior tandem-network grating design inverts component-level spectra through
learned RCWA surrogates; this work is, to our knowledge, the first to invert
a joint display-level specification (MTF, transmission, chromatic spread,
field-edge transmission simultaneously) while anchoring the training loss to
an exact differentiable physics engine whose coupling term is calibrated to
rigorous electromagnetic solutions, and the first neural inverse-design
treatment of PMMA — the lowest-cost, lightest waveguide substrate — under
its full-RGB TIR feasibility window. Every headline design is re-verified
per-polarization with an independent rigorous solver, and the surrogate's
generalization is demonstrated with an explicit memorization audit, both of
which directly answer the reviewer objections that most commonly meet
surrogate-based inverse design.

---

## Where each number comes from (all measured, 2026-07-17)

- Record design: `results/best_design_ever_v3.csv` — J=0.4053, MTF 0.784,
  T 1.25%, chrom 28.8°, at n=1.5 / period 448.1 nm / depth 199.5 nm /
  duty 0.425; rigorous per-pol verification in
  `results/design_rcwa_check_na_v3.csv` (unpol η₁ = 0.096 at 532 nm — 41%
  above the v2 record's actual rigorous coupling).
- Surrogate R²: `results/surrogate_runs_v3.csv` — R²(MTF) 0.99891–0.99935
  over seeds 0–4; all other metrics ≥ 0.9996.
- Tandem: `results/training_runs_v3.csv` — spec-MSE 0.001064
  (surrogate decoder) vs 0.001036 (physics decoder); naive baseline numbers
  from the train_inverse held-out evaluation printout.
- Memorization audit (v3): `results/memorization_audit_v3.csv` — ratio
  1.24×, distance-correlation +0.033.
- [ ] Still to do before submitting: author/affiliation formatting per the
      SPIE portal, and a final read-through against the 250-word limit.
