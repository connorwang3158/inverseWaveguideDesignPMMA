# AR Waveguide Inverse Design — Project Files

Companion project to *Modeling Diffractive Singular Flat AR Waveguide Optical
Performance* (Paper 1). Goal: a machine-learning pipeline in the modern
waveguide inverse-design framework — a **trained forward surrogate network**
that learns the physics, a **tandem inverse network** trained through it, and
a **neural-adjoint design search** that optimizes designs through the trained
network — with every result verified against exact physics (and spot-checked
with rigorous RCWA). See `research_framework.md` for the full publication plan.

**New here? Read `HOW_TO_RUN.md` — the plain-English guide.**

## Files

| File | Purpose |
|---|---|
| `HOW_TO_RUN.md` | Plain-English guide: setup, running, overnight training |
| `research_framework.md` | Publication framework: novelty claim, related work, experiments, venues, timeline |
| `waveguide_physics.py` | Differentiable PyTorch port of Paper 1's forward model (ground truth + data generator) |
| `surrogate.py` | **Forward surrogate network** — learns design→performance from physics-labeled data (Peurifoy 2018 / Liu 2018 framework) |
| `train_inverse.py` | **Tandem inverse network** — trained through the frozen surrogate (default) or the exact physics (ablation) |
| `neural_adjoint.py` | **Neural-adjoint search** — gradient design optimization through the trained surrogate (Ren 2020), physics-verified |
| `optimize_pmma.py`, `sweep_pareto.py` | Non-neural gradient baselines + trade-off frontier |
| `make_3d_model.py` | Interactive 3D waveguide viewer (`waveguide_3d.html`) + STL mesh export |
| `rigorous_solver.py`, `validate.py` | Vectorial RCWA layer (grcwa) + 7-test integrity suite |
| `make_report.py` | Bundles every table/figure into `results_report.html` |
| `overnight.sh` | The full multi-seed training pipeline, unattended |

## Run

```bash
pip install torch numpy matplotlib
python3 waveguide_physics.py            # sanity check: forward pass + gradient check
python3 surrogate.py --pmma --quick     # 1) train the physics-learning network
python3 train_inverse.py --pmma --quick # 2) train the designer through it (tandem)
python3 neural_adjoint.py --quick       # 3) optimize designs through the network
python3 make_3d_model.py                # 4) 3D model of the winning design
bash overnight.sh                       # the real thing (see HOW_TO_RUN.md)
```

## Architecture

```
Stage 1 — learn the physics (surrogate.py):
  design θ (8) ──► ForwardNet ──► spec ŷ (4)     trained on physics-labeled samples

Stage 2 — learn to design (train_inverse.py, tandem):
  target y* (4) ──► InverseNet ──► θ̂ (8) ──► frozen ForwardNet ──► ŷ
                                   loss = ||ŷ − y*||²  (spec space)

Stage 3 — hunt the optimum (neural_adjoint.py):
  θ (trainable) ──► frozen ForwardNet ──► objective J;  ∂J/∂θ through the network

Verification — every stage is scored against the EXACT physics engine
(waveguide_physics.py), and headline designs are spot-checked in RCWA.
```

Spec: [MTF@40cyc/mm, transmission, chromatic spread (°), transmission@20° FOV]
Design: [n, α, σ_RMS, L_corr, thickness, grating period, depth, duty cycle]

## Critical next steps (before publication experiments)

1. **Sync physics constants** — every `# SYNC` comment in `waveguide_physics.py` marks a simplified placeholder. Replace with the exact formulas/values from the Paper 1 GitHub repo, then unit-test until the engine reproduces published numbers (Si₃N₄ loss 93.37–93.39%, PMMA MTF 0.6426–0.6430).
   **Resolved caveat (v2 physics revision):** the design searches originally exploited a missing total-internal-reflection constraint, drifting to large grating periods (~700 nm) whose first order is not actually guided. The v2 engine now enforces the guiding inequality 1 < sinθᵢ + λ/period < n (smooth mask + `tir_penalty()`), restricts the PMMA period bounds to the full-RGB guided window (~430–449 nm), and computes the geometric bounce count — see the `v2 PHYSICS REVISION` block in `waveguide_physics.py`. Both search methods now optimize the same TIR-penalized objective; `neural_adjoint.py` still flags each winner (`tir_guided_green`) as an honesty check, and records from the old engine are quarantined in `archive_old_physics/`.
2. Run the full 5-seed overnight protocol; report surrogate R², tandem spec-MSE (surrogate vs physics decoder), and neural-adjoint vs direct-gradient search in the baseline table (framework §6.3).
3. Out-of-distribution and design-manifold experiments (framework §6.4–6.5).
4. Reverse-engineer published commercial-class specs as the headline case study (§6.6).
5. Spot-check the top recovered designs in `grcwa` (free RCWA) to preempt the "analytic model" review objection.
