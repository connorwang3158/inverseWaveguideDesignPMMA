# AR Waveguide Inverse Design — Project Files

Companion project to *Modeling Diffractive Singular Flat AR Waveguide Optical Performance* (Paper 1). Goal: a physics-anchored tandem neural network that reverse-engineers waveguide designs from target performance specs. See `research_framework.md` for the full publication plan.

## Files

| File | Purpose |
|---|---|
| `research_framework.md` | Publication framework: novelty claim, related work, experiments, venues, timeline |
| `waveguide_physics.py` | Differentiable PyTorch port of Paper 1's forward model (transmission cascade, 5-component MTF, chromatic spread, FOV) |
| `train_inverse.py` | Tandem inverse network + naive baseline, dataset generation, training, evaluation, reverse-engineering demo |
| `inverse_model.pt` | Trained weights (created after running training) |

## Run

```bash
pip install torch
python3 waveguide_physics.py     # sanity check: forward pass + gradient check
python3 train_inverse.py --quick # ~1-2 min smoke test
python3 train_inverse.py         # full run (30k samples, 40 epochs, CPU-ok)
```

## Architecture

```
target spec y* (4) → InverseNet (MLP) → design θ (8) → frozen physics f(θ) → spec ŷ
                                loss = ||ŷ − y*||² (spec space)
```

Spec: [MTF@40cyc/mm, transmission, chromatic spread (°), transmission@20° FOV]
Design: [n, α, σ_RMS, L_corr, thickness, grating period, depth, duty cycle]

## Critical next steps (before publication experiments)

1. **Sync physics constants** — every `# SYNC` comment in `waveguide_physics.py` marks a simplified placeholder. Replace with the exact formulas/values from the Paper 1 GitHub repo, then unit-test until the engine reproduces published numbers (Si₃N₄ loss 93.37–93.39%, PMMA MTF 0.6426–0.6430).
2. Add L-BFGS/random-search and learned-surrogate-tandem baselines (framework §6.3).
3. Out-of-distribution and design-manifold experiments (framework §6.4–6.5).
4. Reverse-engineer published commercial-class specs as the headline case study (§6.6).
5. Optional: spot-check 2–3 recovered designs in `grcwa` (free RCWA) to preempt the "analytic model" review objection.
