# Research Framework: Physics-Anchored Neural Inverse Design of Diffractive AR Waveguides

**Author:** Connor Wang · **Builds on:** *Modeling Diffractive Singular Flat AR Waveguide Optical Performance*
**Target:** Q1/Q2 optics journal · **Date:** July 2026

---

## 1. Working Title

*Physics-Anchored Tandem Neural Networks for Multi-Objective Inverse Design of Singular Flat Diffractive AR Waveguides*

## 2. Research Question

Given a **target system-level performance specification** — system MTF at 40 cyc/mm, total double-coupler transmission, lateral chromatic aberration spread, and FOV — can a neural network **reverse-engineer the waveguide design** (material parameters + grating geometry) that produces it, in milliseconds, without iterative optimization?

## 3. Novelty Claim (the paragraph reviewers will judge)

Prior neural inverse-design work for waveguide gratings inverts a **single component-level metric** — coupling-efficiency spectra of a slanted grating (Optics Express 2024, tandem NN), exit-pupil uniformity for AR-HUD (Applied Optics 2025), or reflection spectra of hybrid gratings (Optics 2025, tandem + cVAE) — and trains on expensive RCWA/FDTD data, so the "forward model" inside the tandem is itself a learned surrogate whose approximation error is never independently measurable. **This work (a) is the first to invert a joint system-level, perceptually-grounded specification (MTF, transmission, chromatic spread, FOV simultaneously) rather than one component metric, and (b) trains the full modern pipeline — forward surrogate network, tandem inverse network through it, and neural-adjoint design search through it — while ALSO holding an exact differentiable analytic engine for the same problem, enabling the first controlled measurement of the surrogate-error contribution** (same tandem trained with learned-surrogate vs. exact-physics decoder; every neural-adjoint winner re-scored by exact physics). The analytic engine is the (validated) forward model from the author's prior paper, reimplemented in PyTorch.

This is the classic strong-sequel structure: Paper 1 built and validated the forward model; Paper 2 learns it with a neural network and inverts it.

## 4. Positioning Against Key Related Work

| Work | What it inverts | Forward model | Gap this paper fills |
|---|---|---|---|
| Tandem NN slanted grating (Opt. Express 32, 12587, 2024) | RGB coupling efficiency | RCWA surrogate | Component-level only; surrogate error |
| AR-HUD uniformity NN (Appl. Opt. 64, 3536, 2025) | Exit-pupil uniformity | NN surrogate (40,000× speedup) | Single metric; no MTF/chromatic |
| Hybrid grating tandem+cVAE (Optics 6(4), 61, 2025) | Reflection spectra | Learned surrogate | Spectra, not display-level metrics |
| End-to-end differentiable geometric waveguide (arXiv 2601.04370, 2026) | Coating stacks, geometric (not diffractive) waveguides | Differentiable ray tracer | Geometric architecture; GPU-workstation scale — no fast inverse network |
| Achromatic metasurface waveguide (Light Sci. Appl. 14, 94, 2025) | Metasurface coupler | Full-wave | No ML inverse; single design |
| **This work** | **Joint MTF + transmission + chromatic + FOV** | **Exact differentiable analytic physics** | — |

## 5. Method: Surrogate-Learned, Physics-Verified Tandem Architecture

```
Stage 1 (surrogate.py) — a network LEARNS the physics:
  design θ (8) ──► Forward Surrogate s(θ) ──► spec ŷ (4)
  trained on physics-labeled samples; accuracy audited per-metric (R², parity)

Stage 2 (train_inverse.py) — a network learns to DESIGN, through Stage 1:
  target spec y* ──► Inverse Network g(y*) ──► θ̂ ──► frozen decoder ──► ŷ
     (4-dim)           (MLP, trainable)       (8-dim)  s(θ̂) [default]     │
                             ▲                         or exact f(θ̂)      │
                             └──────── loss = ‖ŷ − y*‖² (spec space) ◄────┘

Stage 3 (neural_adjoint.py) — designs optimized THROUGH the trained network:
  θ (trainable) ──► frozen s(θ) ──► objective J;  ∂J/∂θ backpropagated
  through the network; finalists re-ranked by exact physics f(θ)

Verification — validation losses and all headline designs scored by the exact
engine f (never by the surrogate's own belief); top designs spot-checked in RCWA.
```

**Design vector θ (8 params):** refractive index n, absorption coefficient α, RMS roughness σ, correlation length Lc, waveguide thickness t, grating period Λ, grating depth d, duty cycle D.

**Spec vector y (4 metrics):** system MTF @ 40 cyc/mm, total transmission, lateral chromatic spread (°, RGB), transmission-at-FOV (evaluated at design FOV).

**Why spec-space loss (tandem trick):** the map y → θ is one-to-many (many designs give the same spec); training θ-space regression directly averages incompatible solutions and fails. Training through the frozen physics decoder sidesteps non-uniqueness — the network only needs to find *a* valid design.

**Differentiable physics engine** (PyTorch port of Paper 1's model): grating equation → diffraction angles per RGB wavelength; Fresnel + Beer-Lambert + Payne-Lacey roughness scattering + grating coupling → transmission cascade; five-component multiplicative MTF cascade (diffraction-limited eye, roughness, chromatic Gaussian PSF, grating, coupler); FOV-transmission scaling. All closed-form ⇒ exact autograd gradients, CPU training in minutes.

## 6. Experiments & Evaluation Plan

1. **Dataset:** sample ~50k θ uniformly within literature-reported bounds (reuse Paper 1's material ranges); compute y = f(θ). Split 80/10/10.
2. **Primary metric:** spec-space reconstruction error |f(g(y*)) − y*| on held-out specs (report per-metric MAE + relative %).
3. **Baselines & ablation:** (a) direct θ-regression MLP (shows non-uniqueness failure), (b) random search / direct gradient ascent on f with equal wall-time budget (`optimize_pmma.py`), (c) tandem with exact-physics decoder vs. the default learned-surrogate decoder (isolates the surrogate-error contribution), (d) neural-adjoint search vs. direct gradient search (does searching through the network find the same optima?).
4. **Out-of-distribution test:** ask for specs *better than any material in the training set* — does the network extrapolate to physically plausible designs or saturate at bounds? (Honest treatment of this earns reviewer trust.)
5. **Design-space analysis:** feed one spec, sample inverse solutions (add input noise), visualize the manifold of equivalent designs — connects to Paper 1's material trade-off discussion.
6. **Reverse-engineering case study:** take published commercial-class waveguide specs (e.g., from Kress & Chatterjee 2020 review) as y*, recover θ̂, compare against known material choices. This is the "reverse engineering" headline demo.
7. **Ablations:** loss weighting across the 4 metrics; with/without bound penalty; dataset size sweep.

## 7. Validation & Integrity Gates

- Round-trip sanity: g(f(θ)) must yield f(g(f(θ))) ≈ f(θ) for training-set θ.
- Physics engine unit tests: reproduce Paper 1's published numbers (e.g., Si₃N₄ 93.37–93.39% loss, PMMA MTF 0.6426–0.6430) before any training.
- State limitations explicitly: analytic model fidelity (not FDTD-validated), independence assumptions in MTF cascade, no fabrication constraints.

## 8. Target Venues (verify current quartiles before submitting)

- **Optics Express** (Optica, Q1, OA, fast) — best fit; two of the closest prior works are here.
- **Applied Optics** (Optica, ~Q2) — strong fallback, AR-HUD inverse paper venue.
- **Optics Continuum / Photonics (MDPI, Q2)** — faster, more forgiving fallbacks.
- Realistic path for a HS-author paper: Optics Express first; cascade down on rejection. Also submit to Regeneron STS / ISEF with the same work.

## 9. Timeline (target submission: Nov–Dec 2026)

| Phase | Weeks | Output |
|---|---|---|
| Physics port + unit tests vs Paper 1 | 1–2 | Verified differentiable engine |
| Dataset + inverse net training | 3–4 | Working model (started this session) |
| Baselines + ablations | 5–7 | Results tables |
| Case study + OOD analysis | 8–9 | Headline figures |
| Writing + internal review | 10–13 | Manuscript |

## 10. Risks

- **"Analytic model too simple" review:** mitigate by framing as rapid design-space exploration (same framing Paper 1 used vs. FDTD), and optionally spot-check 2–3 recovered designs in an open RCWA tool (e.g., `grcwa`, free).
- **Non-uniqueness artifacts:** tandem handles it; cVAE extension is the pre-planned follow-up if reviewers ask for design diversity.
- **Novelty erosion:** field moves fast — re-run the literature scan at writing time.

## 11. Key References Found This Session

- Tandem NN slanted waveguide grating — Optics Express 32, 12587 (2024). https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-7-12587
- Inverse design + uniformity of diffractive waveguides for AR-HUD — Applied Optics 64, 3536 (2025). https://opg.optica.org/ao/abstract.cfm?uri=ao-64-13-3536
- Tandem + cVAE hybrid waveguide gratings — Optics 6(4), 61 (2025). https://doi.org/10.3390/opt6040061
- End-to-end differentiable geometric waveguide design — arXiv:2601.04370 (2026). https://arxiv.org/abs/2601.04370
- Achromatic metasurface waveguide — Light Sci. Appl. 14, 94 (2025). https://www.nature.com/articles/s41377-025-01761-w
- Chromatic aberration & waveguide optimization for full-color AR — Light Adv. Manuf. (2025). https://www.light-am.com/article/doi/10.37188/lam.2025.066
- Inverse design of photonic integrated devices review — APL Photonics 10, 101101 (2025). https://pubs.aip.org/aip/app/article/10/10/101101/3367863
- AutoTandemML (active learning tandem) — arXiv:2502.15643. https://arxiv.org/pdf/2502.15643
- Neural adjoint for meta-optics — arXiv:2604.17425. https://arxiv.org/pdf/2604.17425
