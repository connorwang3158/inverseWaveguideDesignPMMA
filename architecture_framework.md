# Rigorous Claims Framework: Multi-Architecture Waveguide Comparison

Governs what may be claimed about **singular diffractive**, **double stacked diffractive**,
and **geometric (partial-mirror)** waveguides in the paper, at what evidence level, and
with which governing equations. Companion code: `architectures.py`.

---

## 1. The Claim Ladder

Every number in the paper must carry an evidence level. Nothing above its level may be claimed.

| Level | Basis | Permitted claim form |
|---|---|---|
| **L0** | Illustrative model (visualizer defaults) | "qualitative behavior" only — never in Results |
| **L1** | Analytic physics, literature formulas, unit-tested | "model-level comparison under matched assumptions" |
| **L2** | Rigorous solver spot-check (RCWA/TORCWA) | "consistent with rigorous electromagnetic simulation" |
| **L3** | Published experimental data agreement | "validated against reported measurements" |

Current status: singular = **L1** (after `# SYNC` verification against Paper 1);
double & geometric = **L0 in the visualizer**, promoted to **L1 by `architectures.py`** (this update).
Target for the paper: all comparisons at L1, headline designs at L2.

## 2. Governing Equations Per Architecture

### 2.1 Singular flat diffractive (baseline, Paper 1)
As implemented in `waveguide_physics.py`: Fresnel + Beer-Lambert + Tien(1971) roughness
bounce loss + binary-phase-grating scalar efficiency η₁ = 4(sin πD/π)² sin²(φ/2),
φ = 2πd(n−1)/λ; five-component multiplicative MTF; chromatic spread from the in-guide
grating equation n sinθ = λ/Λ. Citable anchors: Goodman (Fourier Optics), Tien 1971,
Payne & Lacey 1994, Watson 2013, Kress & Chatterjee 2020.

### 2.2 Double stacked diffractive (spectral-band split)
Two waveguide layers, layer *i* carrying band Δλᵢ with its own grating tuned to band
center λᵢ (blue+green / red split by default).

- **Chromatic spread per layer:** Δθᵢ evaluated only across Δλᵢ with that layer's Λᵢ —
  *not* a fitted "0.35× factor". Total = max over layers (worst-case eye sees union).
- **Transmission per layer:** singular-model cascade with two extra air-gap interfaces
  (Fresnel) and inter-layer leakage κ (light diffracted by the wrong layer's grating).
  κ is computed from the off-band efficiency of each grating at the other band's
  wavelengths — measurable inside the same scalar model, no free parameter.
- **MTF:** singular cascade per layer; system MTF = luminance-weighted combination.
- Anchor: multi-layer RGB waveguide practice reviewed in Kress & Chatterjee 2020;
  eLight 2023 review (waveguide-based AR displays).

### 2.3 Geometric (partial-mirror array)
In-coupling prism/mirror (no grating), out-coupling by M transflective mirrors.

- **Uniform-extraction design rule** (standard for mirror cascades): mirror k must have
  reflectance R_k = 1/(M+1−k) so each mirror outputs an equal fraction 1/M of the
  guided flux. This is derivable in three lines (include as appendix) — remaining flux
  after k mirrors is (M−k)/M.
- **Transmission:** T = T_fresnel · T_bulk · T_scatter · η_in · [Σ_k out_k · A_k] where
  A_k is the pupil-overlap weight (only mirrors whose exit pupils intersect the eye
  pupil contribute); per-mirror embed/coating loss ε applied M times. Eyebox size and
  brightness trade as 1/M — this coupling must be stated whenever geometric brightness
  is compared to diffractive.
- **Chromatic spread:** mirror reflection is achromatic; residual = coating phase
  dispersion δ(λ), bounded (<0.1°) rather than modeled — claim only the bound.
- **MTF:** no grating modulation terms; add mirror-edge diffraction/step artifact term
  (documented placeholder pending L2) and coating scatter.
- Anchors: Lumus/SCHOTT technical descriptions (design rule), eLight 2023 review;
  patent US11514828 for embed geometry. Note: vendor "10× luminance" claims are
  marketing — never cite as evidence.

## 3. Matched-Assumptions Protocol (what makes the comparison fair)

1. Identical source spectrum (RGB 450/532/635 nm), pupil (3 mm), eye relief, and
   evaluation frequency (40 cyc/mm) across architectures.
2. Identical substrate material model (PMMA parameters from Paper 1 sources).
3. Free parameters optimized *per architecture* with the same optimizer, budget, and
   bounds policy before comparing (compare optima to optima, not defaults to optima).
4. Report per-metric results with the trade-off explicit: geometric gains chromatic
   performance but pays in eyebox-brightness coupling and fabrication constraints —
   state both directions or neither.
5. Multi-seed optimization (≥5 seeds); report best and median ± range. A comparison
   figure without spread bars is not admissible.

## 4. Validation Gates (each gate blocks the claims above it)

- **G1 unit tests:** singular model reproduces Paper 1 published ranges (Si₃N₄ loss
  93.37–93.39%, PMMA MTF 0.6426–0.6430) after `# SYNC` resolution.
- **G2 internal consistency:** double model with both layers assigned the full spectrum
  and κ=0 must reduce exactly to the singular model (limiting-case test, automated).
- **G3 design-rule check:** geometric model with uniform R_k must produce equal
  per-mirror output within numerical tolerance (automated in `architectures.py`).
- **G4 rigorous spot-check (L2):** top-3 optimized designs per diffractive architecture
  re-evaluated in TORCWA/grcwa; report analytic-vs-rigorous deviation.
- **G5 literature sanity:** optimized transmissions must remain below scalar ceiling
  (40.5% per coupler) and within reported ranges for each architecture class.

## 5. Prohibited Claims (reviewer landmines)

- No cross-architecture *manufacturability* or *cost* claims (not modeled).
- No absolute luminance/nits claims (display engine not modeled).
- No "best architecture" conclusion — only per-metric trade-off statements.
- No extrapolation of PMMA results to other materials without rerunning the pipeline.
- Vendor marketing figures (e.g., Lumus 10×) may appear only in Introduction as
  motivation, flagged as unaudited.

## 6. Paper Section Mapping

Methods → §2 equations + §3 protocol. Results → optimized comparisons passing G1–G3
(L1 claims) + G4 table (L2 claims). Discussion → trade-off interpretation + §5 limits.
Appendix → mirror design-rule derivation, limiting-case proofs, unit test outputs.
