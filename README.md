# AR Waveguide Inverse Design — Project Files

Companion project to *Modeling Diffractive Singular Flat AR Waveguide Optical
Performance* (Paper 1). Goal: a machine-learning pipeline in the modern
waveguide inverse-design framework — a **trained forward surrogate network**
that learns the physics, a **tandem inverse network** trained through it, and
a **neural-adjoint design search** that optimizes designs through the trained
network — with every result verified against exact physics (and spot-checked
with rigorous RCWA). See `docs/research_framework.md` for the full
publication plan.

**New here? Read `docs/HOW_TO_RUN.md` — the plain-English guide.**

## Folder map

| Folder | What lives there |
|---|---|
| `docs/` | Instructions & objectives: `HOW_TO_RUN.md` (training instructions), `CHEATSHEET.md`, `research_framework.md` (objectives / publication plan), `CITATIONS.md`, reviews |
| `physics/` | The differentiable physics engine (`waveguide_physics.py`), rigorous RCWA layer, integrity tests, multi-architecture models |
| `networks/` | The neural networks: `surrogate.py` (learns the physics), `train_inverse.py` (tandem designer), `neural_adjoint.py` (record hunter) |
| `baselines/` | Non-neural gradient searches the paper compares against (`optimize_pmma.py`, `sweep_pareto.py`, `optimize_live.py`) |
| `visuals/` | Scripts that build the 3D model and the HTML report, plus the slider playground |
| `figures/` | **Auto-updating charts (PNG)** — loss curves, parity, search, Pareto frontier |
| `results/` | Tables, winners, and the all-time record (`best_design_ever_v2.csv`) + printable STL mesh |
| `checkpoints/` | Trained network weights (`.pt`) |
| `metagrating/` | The separate metagrating/SRG research thread (self-contained) |
| `archive_old_physics/` | Quarantined records from the pre-revision physics (not comparable) |

At the root: `overnight.sh` (the one command), and the two double-click pages
`results_report.html` + `waveguide_3d.html`.

## Run (from the project root)

```bash
pip install torch numpy matplotlib
python3 physics/waveguide_physics.py             # sanity check: forward pass + gradients
python3 networks/surrogate.py --pmma --quick     # 1) train the physics-learning network
python3 networks/train_inverse.py --pmma --quick # 2) train the designer through it (tandem)
python3 networks/neural_adjoint.py --quick       # 3) optimize designs through the network
python3 visuals/make_3d_model.py                 # 4) 3D model of the winning design
bash overnight.sh                                # the real thing (see docs/HOW_TO_RUN.md)
```

## Architecture

```
Stage 1 — learn the physics (networks/surrogate.py):
  design θ (8) ──► ForwardNet ──► spec ŷ (4)     trained on physics-labeled samples

Stage 2 — learn to design (networks/train_inverse.py, tandem):
  target y* (4) ──► InverseNet ──► θ̂ (8) ──► frozen ForwardNet ──► ŷ
                                   loss = ||ŷ − y*||²  (spec space)

Stage 3 — hunt the optimum (networks/neural_adjoint.py):
  θ (trainable) ──► frozen ForwardNet ──► objective J;  ∂J/∂θ through the network

Verification — every stage is scored against the EXACT physics engine
(physics/waveguide_physics.py), and headline designs are spot-checked in RCWA.
```

Spec: [MTF@40cyc/mm, transmission, chromatic spread (°), transmission@20° FOV]
Design: [n, α, σ_RMS, L_corr, thickness, grating period, depth, duty cycle]

## Critical next steps (before publication experiments)

1. **Sync physics constants** — every `# SYNC` comment in `physics/waveguide_physics.py` marks a simplified placeholder. Replace with the exact formulas/values from the Paper 1 GitHub repo, then unit-test until the engine reproduces published numbers (Si₃N₄ loss 93.37–93.39%, PMMA MTF 0.6426–0.6430).
   **Resolved caveat (v2 physics revision):** the design searches originally exploited a missing total-internal-reflection constraint, drifting to large grating periods (~700 nm) whose first order is not actually guided. The v2 engine now enforces the guiding inequality 1 < sinθᵢ + λ/period < n (smooth mask + `tir_penalty()`), restricts the PMMA period bounds to the full-RGB guided window (~430–449 nm), and computes the geometric bounce count — see the `v2 PHYSICS REVISION` block in `waveguide_physics.py`. Both search methods now optimize the same TIR-penalized objective; `neural_adjoint.py` still flags each winner (`tir_guided_green`) as an honesty check, and records from the old engine are quarantined in `archive_old_physics/`.
2. **Scalar-grating validity limit (found by the 2026-07-13 RCWA audit of the record design).** At the TIR-mandated PMMA periods (430–449 nm < λ), scalar diffraction theory overestimates the first-order coupling efficiency badly: at the record design (period 438 nm, depth 400 nm), scalar η₁ = 0.347 at 532 nm vs rigorous RCWA 0.068 unpolarized (~5×), and ~15× at 635 nm — see `results/design_rcwa_check_na.csv` and `metagrating/rcwa_depth_sweep.csv`. RCWA also moves the depth optimum to ~200 nm (rigorous η peaks there; scalar keeps climbing to the 400 nm bound). Consequence: v2-engine *rankings and trends* within the design space remain L1 findings, but absolute transmission and the depth choice must be quoted at the RCWA level; the highest-impact upgrade is an RCWA-calibrated grating-efficiency term (fit a small interpolant/network to a grcwa grid over period × depth × duty × λ × pol) — a v3 physics revision that the checkpoint physics-probe system will automatically propagate through retraining.
3. Run the full 5-seed overnight protocol; report surrogate R², tandem spec-MSE (surrogate vs physics decoder), and neural-adjoint vs direct-gradient search in the baseline table (framework §6.3).
4. Out-of-distribution and design-manifold experiments (framework §6.4–6.5).
5. Reverse-engineer published commercial-class specs as the headline case study (§6.6).
6. Spot-check the top recovered designs in `grcwa` (free RCWA) to preempt the "analytic model" review objection.
