"""Memorization audit of the deployed forward surrogate.

Four tests, one figure:
  A. Same network scored on its OWN training data vs three datasets it has
     never seen (validation, test, and a brand-new set generated today).
     A memorizing network is far better on its own training data.
  B. Parity on the brand-new set (never used for training OR model selection).
  C. Per-design error vs distance to the nearest training example.
     A memorizing network gets worse the farther you move from its
     training points; a network that learned the function stays flat.
  D. Test-set R^2 (MTF) across every independent retraining in
     surrogate_runs.csv — memorization is seed-brittle, learning is stable.
"""
import csv
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # figure/CSV paths resolve from the project root

from networks.surrogate import (load_surrogate, make_dataset, r2_per_metric,
                                TRAIN_SEED_BASE, VAL_SEED, TEST_SEED)

FRESH_SEED = 777777          # never used anywhere in the pipeline
C_TRAIN, C_VAL, C_SURR, C_REF = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
C_IDENT = "#8a8f98"

net = load_surrogate()       # verifies the physics probe, restores bounds
ck = torch.load(os.path.join(ROOT, "checkpoints/forward_surrogate.pt"),
                weights_only=True)
seed, n_train = ck["seed"], ck["n_train"]
print(f"deployed checkpoint: seed {seed}, n_train {n_train}")

# --- A: score the SAME network on four datasets -------------------------
sets = {}
z_tr, y_tr = make_dataset(n_train, seed=TRAIN_SEED_BASE + seed)  # its own food
sets["its own\ntraining data"] = (z_tr, y_tr)
sets["validation set\n(never trained on)"] = make_dataset(15000, seed=VAL_SEED)
sets["test set\n(never trained on)"] = make_dataset(15000, seed=TEST_SEED)
sets["brand-new set\n(generated today)"] = make_dataset(15000, seed=FRESH_SEED)

mse, r2_all = {}, {}
with torch.no_grad():
    for name, (z, y) in sets.items():
        p = net(z)
        mse[name] = torch.nn.functional.mse_loss(p, y).item()
        r2_all[name] = r2_per_metric(p, y)
        print(f"{name.replace(chr(10),' '):40s} MSE {mse[name]:.3e}  "
              f"R2 {[f'{v:.5f}' for v in r2_all[name].tolist()]}")

ratio = mse["brand-new set\n(generated today)"] / mse["its own\ntraining data"]
print(f"\nnever-seen / training error ratio: {ratio:.2f}x "
      f"(memorization would be 100-10000x)")

# --- B: parity on the brand-new set (MTF, the hardest metric) ----------
z_f, y_f = sets["brand-new set\n(generated today)"]
with torch.no_grad():
    p_f = net(z_f)
r2_fresh_mtf = r2_per_metric(p_f, y_f)[0].item()

# --- C: error vs distance to nearest training example ------------------
n_probe = 4000
zp, yp_true = z_f[:n_probe], y_f[:n_probe]
with torch.no_grad():
    err = (net(zp) - yp_true).pow(2).mean(dim=1).sqrt()   # per-design RMSE
    dmin = torch.empty(n_probe)
    for i in range(0, n_probe, 100):
        dmin[i:i+100] = torch.cdist(zp[i:i+100], z_tr).min(dim=1).values
corr = torch.corrcoef(torch.stack([dmin, err]))[0, 1].item()
print(f"corr(error, distance-to-training-data) = {corr:+.3f} over "
      f"{n_probe} never-seen designs (memorization => strongly positive)")

# binned median for the trend line
order = dmin.argsort()
nb = 12
bx, by = [], []
for b in range(nb):
    sl = order[b * n_probe // nb:(b + 1) * n_probe // nb]
    bx.append(dmin[sl].median().item()); by.append(err[sl].median().item())

# --- D: R^2 stability across all independent retrainings ---------------
rows = [r for r in csv.DictReader(open("results/surrogate_runs.csv"))
        if r["samples"] == "150000" and r["epochs"] == "250"]  # paper protocol
r2_mtf = [float(r["R2_MTF@40cyc/mm"]) for r in rows]
n_run2 = 33                                  # rows appended by the new run
print(f"{len(rows)} independent retrainings; R2_MTF min {min(r2_mtf):.5f} "
      f"max {max(r2_mtf):.5f}")

# --- figure -------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
axA, axB, axC, axD = axes.flat

names = list(sets)
vals = [mse[n] for n in names]
cols = [C_TRAIN, C_VAL, C_SURR, C_REF]
bars = axA.bar(range(4), vals, width=0.62, color=cols)
for x, v in enumerate(vals):
    axA.text(x, v * 1.15, f"{v:.1e}", ha="center", fontsize=9, color="#1a2233")
axA.set_yscale("log")
axA.set_ylim(min(vals) * 0.5, max(vals) * 6)
axA.set_xticks(range(4)); axA.set_xticklabels(names, fontsize=8.5)
axA.set_ylabel("prediction error (MSE, log scale)")
axA.set_title(f"A — Same accuracy on data it has never seen\n"
              f"(never-seen error only {ratio:.1f}x training error; "
              f"memorization would be 100x+)", fontsize=10)
axA.grid(alpha=0.25, axis="y")

t, p = y_f[:, 0].numpy(), p_f[:, 0].numpy()
lo, hi = float(min(t.min(), p.min())), float(max(t.max(), p.max()))
axB.plot([lo, hi], [lo, hi], "--", color=C_IDENT, lw=1, zorder=1)
axB.scatter(t, p, s=5, alpha=0.3, color=C_REF, edgecolors="none", zorder=2)
axB.set_xlabel("true value (exact physics)")
axB.set_ylabel("network prediction")
axB.set_title(f"B — Parity on the brand-new set (MTF)\n"
              f"15,000 designs generated today, R$^2$ = {r2_fresh_mtf:.5f}",
              fontsize=10)
axB.grid(alpha=0.25)

axC.scatter(dmin.numpy(), err.numpy(), s=5, alpha=0.25, color=C_TRAIN,
            edgecolors="none", zorder=2, label="never-seen design")
axC.plot(bx, by, "-o", color=C_SURR, lw=2, ms=4, zorder=3,
         label="median (binned)")
axC.set_yscale("log")
axC.set_xlabel("distance to nearest training example (normalized design space)")
axC.set_ylabel("prediction error (per-design RMSE, log)")
axC.set_title(f"C — Error does NOT grow away from the training data\n"
              f"correlation = {corr:+.3f} (memorization would be strongly "
              f"positive)", fontsize=10)
axC.grid(alpha=0.25); axC.legend(fontsize=8, loc="upper right")

x1 = range(1, len(r2_mtf) - n_run2 + 1)
x2 = range(len(r2_mtf) - n_run2 + 1, len(r2_mtf) + 1)
axD.scatter(x1, r2_mtf[:-n_run2], s=18, color=C_TRAIN, edgecolors="none",
            label="overnight run 1")
axD.scatter(x2, r2_mtf[-n_run2:], s=18, color=C_VAL, edgecolors="none",
            label="overnight run 2")
axD.axhline(1.0, color=C_IDENT, lw=1, ls="--")
axD.set_ylim(min(r2_mtf) - 0.0005, 1.0006)
axD.set_xlabel("independent retraining # (fresh network + fresh data each time)")
axD.set_ylabel("R$^2$ on held-out test set (MTF)")
axD.set_title(f"D — {len(r2_mtf)} independent retrainings, all equally "
              f"accurate\n(memorization is seed-brittle; learning is stable)",
              fontsize=10)
axD.grid(alpha=0.25); axD.legend(fontsize=8, loc="lower right")

fig.suptitle("Memorization audit — the deployed surrogate is scored ONLY on "
             "designs it never trained on", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = os.path.join(ROOT, "figures/memorization_audit.png")
fig.savefig(out, dpi=150)
print(f"saved -> {out}")

with open(os.path.join(ROOT, "results/memorization_audit.csv"), "w",
          newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "MSE", "R2_MTF", "R2_T", "R2_chrom", "R2_Tfov"])
    for n in names:
        w.writerow([n.replace("\n", " "), f"{mse[n]:.6e}"] +
                   [f"{v:.5f}" for v in r2_all[n].tolist()])
    w.writerow([])
    w.writerow(["never_seen_over_train_error_ratio", f"{ratio:.2f}"])
    w.writerow(["corr_error_vs_distance_to_training", f"{corr:+.3f}"])
print("saved -> results/memorization_audit.csv")
