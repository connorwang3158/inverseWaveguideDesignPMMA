# Training Cheat Sheet — AR Waveguide Inverse Design

Everything you need to run, watch, and modify training. Keep this open next to Terminal.

---

## 1. How the code works (the 30-second mental model)

```
physics/waveguide_physics.py   the SIMULATOR (the "world")
      theta (8 numbers: material + geometry)  ──►  y (4 scores: MTF, T, chrom, T@FOV)

networks/surrogate.py          the STUDENT (a neural net that LEARNS the simulator)
networks/train_inverse.py      the TRANSLATOR (spec ──► design), learned through the student
networks/neural_adjoint.py     the TREASURE HUNTER (searches through the trained network)
baselines/optimize_pmma.py     the RIVAL (searches the physics directly, for comparison)
baselines/sweep_pareto.py      the MENU MAKER (best design under 6 different priorities)
physics/architectures.py       double-layer & mirror-based waveguide models + self-tests
visuals/waveguide_visualizer.html   the PLAYGROUND (double-click, drag sliders)
```

**How training works, one sentence per step:**
1. Make 30,000 random waveguide designs (`sample_theta`).
2. Score them all with the physics simulator (`forward_model`) → the dataset.
3. The neural net (`InverseNet`, an 4→256→256→256→256→8 MLP) sees the scores and
   guesses the designs.
4. Each guess is pushed BACK through the physics; the loss is how far the guessed
   design's *scores* land from the requested scores ("tandem" trick — this is what
   handles many-designs-one-spec ambiguity).
5. `loss.backward()` computes, via calculus, how to nudge all ~200k network weights;
   `opt.step()` applies the nudge. Repeat per batch, per epoch.
6. Every epoch it's graded on 3,000 designs it never trained on (validation) —
   the number that matters.

## 2. Run commands (copy-paste)

```bash
cd <your project folder>                     # then, from the project root:

python3 physics/waveguide_physics.py         # sanity check the simulator
python3 networks/surrogate.py --pmma --quick # 1-min smoke test (physics learner)
python3 networks/train_inverse.py --pmma --quick   # 1-min smoke test (designer)
python3 networks/train_inverse.py --pmma --samples 100000 --epochs 300 --batch 256 --seed 0
                                             # rigorous run: 117,300 iterations
python3 networks/neural_adjoint.py           # hunt for records through the network
python3 baselines/optimize_pmma.py           # direct-search rival
python3 baselines/sweep_pareto.py            # trade-off menu + chart
python3 physics/architectures.py             # double/geometric models + gate tests
open figures/loss_curve.png                  # view the training curve
```

Paper protocol: repeat the rigorous run with `--seed 0` through `--seed 4` (5 runs),
report median ± range of "best validation spec-MSE".

## 3. Reading the training printout

```
ep  12/40 | train loss 0.00092 | val spec-MSE 0.00089 | baseline theta-MSE 0.058 | 7s
    │            │                    │                      │
    epoch        error on data        error on UNSEEN data   the naive rival
                 it trains on         (the score that counts) (should stay bad)
```

- Both falling → learning. ✔
- Train falls, val rises → overfitting (the saved checkpoint auto-protects you).
- Both flat from the start → learning rate too low or a bug.
- Val bounces wildly → learning rate too high; try `--lr 3e-4`.

## 4. Knobs you can turn WITHOUT editing code

| Flag | What it does | Try when |
|---|---|---|
| `--samples 100000` | bigger dataset | val error plateaus too high |
| `--epochs 300` | train longer | curve still falling at the end |
| `--batch 256` | smaller batches = more iterations, noisier steps | chasing the last bit of accuracy |
| `--lr 3e-4` | smaller learning steps | val loss bounces around |
| `--seed 3` | different random start | always, ×5 for the paper |

Change ONE thing per run, compare `figures/loss_curve.png` before/after. That's the science.

## 5. Edits INSIDE the code (open the .py file in a text editor)

**Change what "optimal" means** — `optimize_pmma.py`, near the top:
```python
W_MTF, W_T, W_CA = 1.0, 1.0, 0.5   # sharpness, brightness, color-fringing penalty
```
Make brightness king: `0.3, 3.0, 0.5`. Hate color fringing: `1.0, 1.0, 3.0`.

**Change the wish in the demo** — `train_inverse.py`, in `demo_reverse_engineering`:
```python
y_star = torch.tensor([[0.65, 0.065, 14.0, 0.045]])
#                        MTF    T   chrom°  T@FOV
```

**Change PMMA's allowed ranges** — `waveguide_physics.py`, `PMMA_BOUNDS`:
```python
[300.0, 700.0],    # period (nm)  <- widen/narrow the search space here
```

**Make the network bigger** — `train_inverse.py`, `InverseNet()`:
```python
def __init__(self, hidden: int = 256, depth: int = 4):   # try hidden=512, depth=6
```
Bigger = slower + more data-hungry; only helps if val error is the bottleneck.

**Change design priorities the net trains toward** — `SPEC_SCALE` in
`waveguide_physics.py` balances the 4 metrics in the loss; halving a value makes
errors in that metric count 4× more.

## 6. The `# SYNC` job (your #1 task before paper numbers)

Search `physics/waveguide_physics.py` for `# SYNC`. Each marks a constant I simplified.
For each: find the exact value/formula in your Paper 1 repo or its cited source,
replace, rerun `python3 physics/waveguide_physics.py`, and check numbers still look sane.
Done when the model reproduces Paper 1: PMMA loss 93.64–94.10%, MTF 0.6426–0.6430.

## 7. When things break

| Error | Fix |
|---|---|
| `No module named torch` | `pip3 install torch` |
| `No module named numpy/matplotlib` | `pip3 install numpy matplotlib` |
| `No such file or directory` on cd | drag the folder into Terminal after typing `cd ` |
| git "index.lock exists" | delete `.git/index.lock` in Finder (Cmd+Shift+. shows hidden files) |
| anything else | paste the full error to Claude |
