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

## 2. The five-minute tour (run these in order)

```bash
python3 waveguide_physics.py          # sanity-check the physics simulator
python3 surrogate.py --pmma --quick   # train the physics-learning network (~1 min)
python3 train_inverse.py --pmma --quick   # train the designer network (~2 min)
python3 neural_adjoint.py --quick     # let the network hunt for the best design
python3 make_3d_model.py              # build the 3D model of the winner
python3 make_report.py                # bundle everything into one web page
```

Then double-click **`results_report.html`** (all tables + charts) and
**`waveguide_3d.html`** (rotate the waveguide in 3D, drag the sliders).

`--quick` runs are small smoke tests. Drop the flag for real runs.

## 3. Training overnight (the real runs)

One command starts the whole pipeline — 5 surrogate seeds, 5 inverse-network
seeds, the ablation run, the design searches, and all figures:

```bash
bash overnight.sh
```

- Keep the laptop **plugged in** with the **lid open**. On a Mac the script
  automatically uses `caffeinate` so the machine won't fall asleep.
- It is a LONG night — the 5+5 seed protocol can take 10+ hours on a laptop
  CPU. If morning arrives first, it's fine: every finished seed is already
  saved. You can also shorten the night by editing the `--epochs` or the seed
  lists at the top of `overnight.sh`.
- Watch live progress from a second terminal tab:

  ```bash
  tail -f overnight_log.txt
  ```

- In the morning, open `results_report.html`. Everything is also saved as
  CSVs, so nothing is lost if you close the page.

If the run is interrupted, just run `bash overnight.sh` again — checkpoints
named `*_seed0.pt` etc. are kept per run, and the best-ever files
(`forward_surrogate.pt`, `best_design_ever.csv`) only update when a new run
actually beats the record.

## 4. What each file means

| You run | You get | What it is |
|---|---|---|
| `surrogate.py` | `forward_surrogate.pt`, `surrogate_loss_curve.png`, `surrogate_parity.png`, `surrogate_runs.csv` | The trained physics-emulator network + proof of how well it learned (parity plot: predictions vs truth; R² near 1.0 = learned) |
| `train_inverse.py` | `inverse_model_seed*.pt`, `loss_curve.png`, `training_runs.csv` | The trained designer network + its learning curve |
| `neural_adjoint.py` | `optimal_designs_na.csv`, `neural_adjoint_run.png`, `best_design_ever.csv` | Best designs found by searching through the network, verified with exact physics |
| `optimize_pmma.py`, `sweep_pareto.py` | `optimal_designs.csv`, `pareto_*.{csv,png}` | Non-neural baselines the paper compares against |
| `make_3d_model.py` | `waveguide_3d.html`, `waveguide_model.stl` | Interactive 3D model + a mesh file any 3D viewer/printer opens |
| `make_report.py` | `results_report.html` | One page with every table and figure |
| `validate.py` | pass/fail printout | 7 physics integrity tests (needs `grcwa`) |

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
range of "best validation spec-MSE" from `training_runs.csv` /
`surrogate_runs.csv`.

## 7. When things break

| Error | Fix |
|---|---|
| `No module named torch` | `pip3 install torch` |
| `No module named numpy/matplotlib` | `pip3 install numpy matplotlib` |
| `forward_surrogate.pt not found` | run `python3 surrogate.py --pmma` first — the other scripts need the trained surrogate |
| `No such file or directory` on `cd` | type `cd `, drag the project folder into the terminal, Enter |
| 3D page is black | it needs internet once to fetch the graphics library; offline, open `waveguide_model.stl` instead |
| Overnight run died partway | rerun `bash overnight.sh` — records only improve, never regress |
