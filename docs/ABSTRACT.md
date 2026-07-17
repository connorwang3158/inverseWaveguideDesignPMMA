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
- Numbers marked ⟨v3⟩ come from the first single-seed v3 run (2026-07-17);
  refresh them after the first full 5-seed v3 overnight before circulating.

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
R² ≥ 0.998 across all metrics and passes a four-test memorization audit; a
tandem inverse network trained through the frozen forward model resolves the
one-to-many inverse problem, recovering designs that meet requested
specifications in milliseconds; and a neural-adjoint search over the trained
surrogate discovers record designs. Rigorous calibration qualitatively
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

## Fill-in checklist before submission

- [ ] ⟨v3⟩ record depth/MTF/T from `results/best_design_ever_v3.csv` after the
      first full overnight (5 seeds; the single-seed 2026-07-17 numbers are
      placeholders of the same magnitude).
- [ ] Surrogate R² under v3 from `results/surrogate_runs_v3.csv` (quote the
      5-seed range, not one seed).
- [ ] Tandem spec-MSE under v3 from `results/training_runs_v3.csv`.
- [ ] Verify author/affiliation formatting per SPIE portal requirements.
