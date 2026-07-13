# Spec Review — "High-Rigor Optimization of a PMMA Slanted-SRG In-Coupler"

**Reviewed:** 2026-07-12 · **Verdict: sound and defensible — approve with 6
amendments.** The narrow claim, RCWA core, UQ, and reviewer-criticism mapping
are exactly right. Every amendment below is grounded in numbers computed today
in this repo (physics-v2 engine + grcwa runs; see `PHYSICS_VALIDATION.md`,
`rcwa_depth_sweep.csv`, `design_rcwa_check.csv`).

---

## A. Confirmations (today's runs already de-risk parts of the spec)

1. **grcwa is a viable engine and passes the spec's own gates.** 9/9
   validation tests: energy conservation |R+T−1| < 2.5·10⁻¹⁴ at every
   evaluation, Fourier-order convergence spread 3.8·10⁻⁶ over nG = 41–81,
   Fresnel exactness to 3.5·10⁻¹², ±1 parity symmetry to 1.8·10⁻¹⁵
   (`validate.py`). The §5.6 convergence study is already scripted
   (`convergence_check()`); the slanted profile (staircase) is already
   implemented (`profile_slanted` in `rigorous_solver.py`).
2. **The scalar→vector gap the spec is built on is real and large.** At
   Λ = 438 nm / 532 nm on PMMA, scalar theory says depth 400 nm, η = 34.6%;
   RCWA says the true unpolarized optimum is depth ≈ 200 nm, η = 8.5%
   (binary profile). A ~5× overestimate — the strongest possible motivation
   for §5.2, quotable in the introduction.
3. **Polarization splitting is first-order, not a footnote.** TE/TM = 4.2×
   at the binary optimum (13.8% vs 3.3%). Optimizing TE-only (polarized
   projector) is a legitimate primary objective; report both (§5.3 ✓).

## B. Amendments

1. **§5.3 — Don't fully pin the period; make it a bounded variable (the spec's
   own "optionally").** For 532 nm the guided window is λ/n < Λ < λ →
   **357–532 nm**; the choice of θ_d within (asin(1/n), 90°) is a real design
   trade (steeper θ_d → fewer bounces & more FOV headroom on one side, but
   longer path and tighter fabrication). Pinning θ_d to one value imports an
   arbitrary constant into the headline claim. Optimize {φ, F, d, Λ} with
   Λ ∈ [370, 500] nm and report the optimum's θ_d. Cost: one extra dimension
   on a problem you can already afford to solve exactly thousands of times.
2. **§5.4 — State the ±1-order asymmetry ratio as a secondary metric.** The
   whole point of slant is breaking parity (our unslanted binary gives exactly
   symmetric ±1, verified to 1.8·10⁻¹⁵). Reporting η₋₁/η₊₁ quantifies how much
   of the gain comes from redirecting the wrong-direction order — reviewers
   will ask.
3. **§5.6 — Add the re-interaction caveat to §7, cite Zhao et al. 2024
   (Opt. Express 32, 12340).** Single-pass RCWA η is an upper bound on system
   in-coupling: guided light re-hits the in-coupler and partially out-couples.
   Zhao's reciprocity/energy bound (with thickness, pupil size, relief
   distance) is the right frame; one paragraph plus a bound estimate for the
   winning geometry retires the objection cheaply. (Our t ≈ 2 mm optimum from
   the system-level engine is favorable here: thicker slab → fewer
   re-interactions.)
4. **§5.3 — n_PMMA at 532 nm ≈ 1.4936 (dispersion), not 1.49 flat.** Use a
   cited Sellmeier/Cauchy fit for optical-grade PMMA and propagate the grade
   range through UQ as planned (Nilsen et al. 2025 is the AR-context anchor;
   add a primary dispersion source for the fit itself).
5. **§5.5 — Seed-fixed differential evolution is right; add a NG≥81
   production setting.** Our convergence data shows nG = 41 already stable to
   1e-5 for binary PMMA at these periods, but slanted staircases (16+
   sublayers) converge slower in the groove direction — run the convergence
   study *on a slanted structure*, not the binary one, before fixing nG.
6. **§5.6 validation anchor — the Ansys n=1.8 case checks the solver, not the
   claim.** Fine as solver validation; pair it with a published *low-index or
   polymer* SRG result if one exists (even qualitative) so the PMMA envelope
   isn't extrapolated entirely from a high-index anchor. If none exists,
   state that explicitly — it strengthens the novelty claim.

## C. Relationship to the existing codebase

The spec's project is the **L2 core** of the existing claims ladder — it
replaces the coupler-level scalar formula; the system-level cascade
(`waveguide_physics.py` v2: TIR window, polarized Fresnel, Tien roughness,
bounce geometry, MTF) remains the L1 envelope that turns coupler η into
display metrics, and the index-ceiling analysis (§5.8) is already computable:
`fov_window_deg()` gives the exact index-limited field window (4.4° full-RGB
at Λ=438; single-λ windows are wider). The tandem-NN work becomes the spec's
§11 Approach B, verbatim.

## D. Assumptions to confirm (§12) — answered from this repo

- 532 nm single-λ: consistent with everything here; the RGB TIR window
  (429–450 nm) analysis stays as motivation for why single-λ is the honest
  scope for PMMA.
- Air-clad SRG on bulk PMMA: matches `rigorous_solver.py` geometry
  (air / corrugation staircase / semi-infinite PMMA) exactly.
- In-coupler only with out-coupler symmetry deferred: fine — note that
  out-coupling occurs at the *guided* incidence angle, so symmetry is an
  approximation to flag, not assume silently.
