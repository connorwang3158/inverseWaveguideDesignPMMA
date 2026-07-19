# How to Run This Project (Plain-English Guide)

This project trains **two neural networks** that learn to design PMMA
(acrylic plastic) AR waveguides:

1. **The forward surrogate** (`surrogate.py`) — a network that *learns the
   physics*. You show it hundreds of thousands of (design → performance)
   examples until it can predict a waveguide's optical performance on its own.
2. **The inverse network** (`train_inverse.py`) — a network that *designs*.
   You give it a performance wish-list (sharpness, brightness, low color
   fringing) and it answers with the waveguide geometry that achieves it.
   It learns by back-propagating through the frozen surrogate network — the
   "tandem" architecture used by modern waveguide inverse-design papers.
3. **The neural-adjoint search** (`neural_adjoint.py`) then hunts for the
   *best possible* PMMA design by running gradient ascent **through the
   trained network**. Every candidate is double-checked against the exact
   physics equations before it counts.

---

## 1. One-time setup (5 minutes)

You need Python 3 (macOS already has it; on Windows install from python.org
and tick "Add to PATH"). Then, in a terminal, from this project folder:

```bash
pip3 install torch numpy matplotlib
```

Optional (only for the rigorous Maxwell-equation checks):

```bash
pip3 install grcwa autograd
```

To open a terminal *in this folder*: on macOS, open Terminal, type `cd `
(with a space), drag the project folder onto the window, press Enter.

## 2. The five-minute tour (run these in order, from the project root)

```bash
python3 physics/waveguide_physics.py           # sanity-check the physics simulator
python3 networks/surrogate.py --pmma --quick   # train the physics-learning network (~1 min)
python3 networks/train_inverse.py --pmma --quick   # train the designer network (~2 min)
python3 networks/neural_adjoint.py --quick     # let the network hunt for the best design
python3 visuals/make_3d_model.py               # build the 3D model of the winner
python3 visuals/make_report.py                 # bundle everything into one web page
```

Then double-click **`results_report.html`** (all tables + charts) and
**`waveguide_3d.html`** (rotate the waveguide in 3D) — both sit at the
project root.

`--quick` runs are small smoke tests. Drop the flag for real runs.

## 3. Training overnight (the real runs — guaranteed 12+ hours)

> **v5 note (2026-07-18):** before an overnight run, read
> `docs/OVERNIGHT_V5.md` — it has the pre-flight checks, the expected v5
> number ranges (MTF ≈ 0.44–0.47, throughput FOM ≈ 1.0–1.4%, walk-off
> ≈ 1.5–2.5 mm), and the morning paper-resync checklist.

One command starts the whole pipeline and **keeps working until at least 12
hours have passed**:

```bash
bash overnight.sh
```

What the night looks like (these markers appear in `overnight_log.txt`):

```
stage 1/6: surrogate seed 0..4       train the physics-learning network, 5 seeds
stage 2/6: inverse ... seed 0..4     train the designer network THROUGH it, 5 seeds
stage 3/6: ablation                  same training through exact physics (paper table)
stage 4/6: design search             neural adjoint + gradient/Pareto baselines
stage 5/6: figures, 3D model, report
stage 6/6: bonus lap, seed 5, 6, …   one extra seed + record hunt per lap,
                                     repeating until the 12 h budget is spent
```

- Keep the laptop **plugged in** with the **lid open**. On a Mac the script
  automatically uses `caffeinate` so the machine won't fall asleep.
- Want a different night length? `HOURS=16 bash overnight.sh` (or `HOURS=0`
  for the core protocol only, no bonus laps).
- Watch live progress from a second terminal tab (the first tab stays quiet
  on purpose — everything is written to the log):

  ```bash
  tail -f overnight_log.txt
  ```

- New records print `NEW RECORD` in the log and land in
  `results/best_design_ever_v2.csv` (stage 4 onward, then once per bonus lap).
- In the morning, open `results_report.html` — it is rebuilt after every
  bonus lap, so it is always current. Everything is also saved as CSVs, so
  nothing is lost if you close the page.

If the run is interrupted, just run `bash overnight.sh` again — per-seed
checkpoints are kept, and the best-ever files (`checkpoints/forward_surrogate.pt`,
`results/best_design_ever_v2.csv`) only update when a new run actually beats
the record.

**If you edit `waveguide_physics.py`**, the pipeline notices by itself: every
surrogate checkpoint stores a fingerprint of the physics it learned from, and
the next training run replaces any checkpoint whose fingerprint no longer
matches (scripts that need the surrogate refuse stale checkpoints with a clear
message instead of silently using them). Records from a previous physics
engine belong in `archive_old_physics/`, not mixed into the live tables.

## 4. What each file means

| You run | You get | What it is |
|---|---|---|
| `networks/surrogate.py` | `checkpoints/forward_surrogate.pt`, `figures/surrogate_loss_curve.png`, `figures/surrogate_parity.png`, `results/surrogate_runs.csv` | The trained physics-emulator network + proof of how well it learned (parity plot: predictions vs truth; R² near 1.0 = learned) |
| `networks/train_inverse.py` | `checkpoints/inverse_model_seed*.pt`, `figures/loss_curve.png`, `results/training_runs.csv` | The trained designer network + its learning curve |
| `networks/neural_adjoint.py` | `results/optimal_designs_na.csv`, `figures/neural_adjoint_run.png`, `results/best_design_ever_v2.csv` | Best designs found by searching through the network, verified with exact physics |
| `baselines/optimize_pmma.py`, `baselines/sweep_pareto.py` | `results/optimal_designs.csv`, `results/pareto_results.csv`, `figures/pareto_front.png` | Non-neural baselines the paper compares against |
| `visuals/make_3d_model.py` | `waveguide_3d.html` (root), `results/waveguide_model.stl` | Interactive 3D model + a mesh file any 3D viewer/printer opens |
| `visuals/make_report.py` | `results_report.html` (root) | One page with every table and figure |
| `physics/validate.py` | pass/fail printout | 7 physics integrity tests (needs `grcwa`) |

## 5. How to read the training printout

```
ep  12/40 | train loss 0.00092 | val spec-MSE 0.00089 | surr-val 0.00071 | ...
```

- **train loss** — error on examples the network is learning from.
- **val spec-MSE** — error on designs it has NEVER seen, scored by the *exact
  physics*. **This is the number that counts.**
- **surr-val** — what the surrogate network *believes* the error is. If this
  is much lower than val spec-MSE, the surrogate is flattering itself —
  train the surrogate longer or with more samples.
- Both falling → learning. Train falls while val rises → overfitting (the
  saved checkpoint auto-protects you). Val bouncing wildly → lower the
  learning rate (`--lr 3e-4`).

## 6. Training it yourself — the knobs

Every training script takes the same flags; change ONE per run and compare
the loss-curve PNGs before/after:

| Flag | Meaning | Try when |
|---|---|---|
| `--samples 100000` | bigger training dataset | val error plateaus too high |
| `--epochs 300` | train longer | curve still falling at the end |
| `--batch 256` | more, noisier updates | chasing the last bit of accuracy |
| `--lr 3e-4` | gentler learning steps | val loss bounces around |
| `--seed 3` | different random start | always — run seeds 0–4 for the paper |
| `--decoder physics` | (inverse net only) train through exact equations instead of the surrogate | the ablation for the paper |

Paper protocol: run each experiment with seeds 0–4 and report the median ±
range of "best validation spec-MSE" from `results/training_runs.csv` /
`results/surrogate_runs.csv`.

## 7. When things break

| Error | Fix |
|---|---|
| `No module named torch` | `pip3 install torch` |
| `No module named numpy/matplotlib` | `pip3 install numpy matplotlib` |
| `forward_surrogate.pt not found` | run `python3 networks/surrogate.py --pmma` first — the other scripts need the trained surrogate |
| `trained under a DIFFERENT waveguide_physics.py` | the physics engine changed since the surrogate was trained; retrain with `python3 networks/surrogate.py --pmma` (the overnight script does this automatically) |
| `rcwa_eta_grid.npz not found` | the v3 engine's rigorous coupling table is missing from `physics/` (it ships with the repo); rebuild it once with `pip3 install grcwa` then `python3 physics/calibrate_rcwa.py` (~10 min), and expect to retrain afterwards |
| `No such file or directory` on `cd` | type `cd `, drag the project folder into the terminal, Enter |
| 3D page won't load | the viewer is fully self-contained (no internet needed) — try another browser, or open `waveguide_model.stl` in any 3D viewer instead |
| Overnight run died partway | rerun `bash overnight.sh` — records only improve, never regress |
