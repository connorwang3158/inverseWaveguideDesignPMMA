# AR Waveguide Inverse Design — Project Files

Companion project to *Modeling Diffractive Singular Flat AR Waveguide Optical
Performance* (Paper 1). Goal: a physics-anchored tandem neural network that
reverse-engineers waveguide designs from target performance specs, now scoped
per the 2026-07-12 design spec toward a rigorous RCWA-optimized PMMA in-coupler
(see `SPEC_REVIEW.md`). Publication plan: `research_framework.md`.

**Physics v2 (2026-07-12):** the forward engine was audited and revised — TIR
guiding constraint (critical fix), field-angle grating equation,
polarization-resolved (TE/TM) Fresnel, geometric bounce count, exact 3-line
chromatic MTF, ad-hoc floors removed. Full derivations, audit trail, and RCWA
verification: **`PHYSICS_VALIDATION.md`**. All pre-v2 result CSVs were
regenerated; pre-v2 numbers are not comparable (they violated TIR).

## Files

| File | Purpose |
|---|---|
| `research_framework.md` | Publication framework: novelty claim, related work, experiments, venues |
| `architecture_framework.md` | Claims ladder (L0–L3) + per-architecture governing equations |
| `PHYSICS_VALIDATION.md` | **v2 physics derivations, audit findings, RCWA verification** |
| `SPEC_REVIEW.md` | Review of the slanted-SRG in-coupler spec (Approach A) |
| `CITATIONS.md` | Verified sources (audit 2026-07-12) |
| `waveguide_physics.py` | Differentiable forward engine v2 (TIR window, TE/TM, FOV window) |
| `rigorous_solver.py` | grcwa RCWA layer: vector TE/TM, profiles, `--designs` verification |
| `validate.py` | 9 independent tests (energy, parity, Brewster, TIR, convergence…) |
| `optimize_pmma.py` | Multi-start gradient search for best PMMA designs (TIR-penalized) |
| `sweep_pareto.py` | Objective-weight sweep → trade-off frontier |
| `train_inverse.py` | Tandem inverse network + naive baseline |
| `architectures.py` | L1 models: double-stacked diffractive + geometric mirror waveguide |
| `make_3d_viz.py` → `waveguide_designs_3d.html` | **Interactive 3D view of top designs** (Three.js; TE/TM toggle, field-angle slider, live physics, RCWA panel) |
| `waveguide_visualizer.html` | 2D slider explorer (physics synced to v2) |
| `make_report.py` → `results_report.html` | Compile all CSVs/figures into one report |
| `overnight.sh` | 5-seed full training + optimization + report |

## Run

```bash
pip install torch grcwa
python3 waveguide_physics.py     # engine smoke test + TIR sanity check
python3 validate.py              # 9/9 independent physics tests
python3 optimize_pmma.py         # best designs -> optimal_designs.csv
python3 rigorous_solver.py --designs   # RCWA TE/TM check of the winners
python3 make_3d_viz.py           # -> waveguide_designs_3d.html (open in browser)
python3 train_inverse.py --quick --pmma  # ~1-2 min tandem smoke test
bash overnight.sh                # full 5-seed protocol (run AFTER the above pass)
```

## Architecture

```
target spec y* (4) → InverseNet (MLP) → design θ (8) → frozen physics f(θ) → spec ŷ
                                loss = ||ŷ − y*||² (spec space)
```

Spec: [MTF@40cyc/mm, transmission, chromatic spread (°), transmission@FOV]
Design: [n, α, σ_RMS, L_corr, thickness, grating period, depth, duty cycle]

## Headline v2 results (PMMA mode)

- Guided-window constraint: full-RGB requires **Λ ∈ (429, 450) nm**; the
  common RGB field window is only **≈4.4°** — the quantitative index-ceiling
  story (Kress & Chatterjee 2021).
- Optimizer winner (L1): Λ=438 nm, depth 400 nm, duty 0.5, t≈2 mm → T≈11%,
  MTF 0.709, chromatic 31.9°.
- RCWA (vector, L2) correction: scalar overestimates η by ~5× at Λ≈0.8λ;
  true unpolarized optimum **depth ≈ 200 nm** (η=8.5%; η_TE=13.8%, TE/TM≈4).
  Headline designs must quote the RCWA column (`design_rcwa_check.csv`,
  `rcwa_depth_sweep.csv`).

## Metagrating study (spec Approach A+, locked decisions D1–D4)

| File | Purpose |
|---|---|
| `metagrating_model.py` | Differentiable RCWA forward model: ρ(x,z) multilayer, exact no-overhang parameterization, filter + WLS projection, FD-gradient + convergence checks |
| `optimize_metagrating.py` | Two arms: SRG (DE + polish) vs freeform topology (Adam, β-ramp, robust triple). `--smoke` for quick runs; full run overnight |
| `metagrating_dashboard.py` | Live auto-refreshing `metagrating_live.html` + per-metric hall-of-fame (`metagrating_hof.json`) that self-updates across all runs |
| `crosscheck_meent.py` | Cross-solver validation grcwa ↔ meent (spec §5.6.2) |
| `verify_srg_winner.py` | nG-convergence + meent + staircase-resolution check of the SRG winner |

Smoke-run status (2026-07-12, sandbox): FD gradients match autograd to 2e-5;
energy |R+T−1| ~1e-14 every solve; grcwa↔meent agree to ~2e-5; **SRG winner
η_TE(−1) = 92.7%** (slant −45.3°, duty 0.62, depth 381 nm, Λ=429 nm; converged
nG=41→161, meent-confirmed, staircase-stable at NL=48; TM 6.7%). Topology arm
at smoke budgets: 22.5% — production budgets (300 iters × 3 starts × 5 periods,
incl. Λ≈430) required for the real comparison; its design space contains the
SRG by construction (D1). Caveats: single-pass η is an upper bound on system
in-coupling (Zhao 2024 re-interaction); slant/depth NIL bounds pending
literature citation; robust-worst reported alongside every topology result.

Production run: `python3 optimize_metagrating.py` (leave `metagrating_live.html`
open in a browser — it self-refreshes with the current design, efficiency
history, and hall-of-fame).

## Next steps (per SPEC_REVIEW.md)

1. Implement the slanted-SRG RCWA optimization (spec Approach A) with
   Λ ∈ [370, 500] nm as a bounded variable, ±1 asymmetry reporting, and
   slanted-profile convergence study.
2. Cross-solver check (meent) + published reference case.
3. UQ: propagate n_PMMA(λ) dispersion + fabrication tolerances.
4. Sync remaining `# SYNC` L1 heuristics against Paper 1's repo.
5. Overnight 5-seed tandem protocol on the v2 engine (`overnight.sh`).
