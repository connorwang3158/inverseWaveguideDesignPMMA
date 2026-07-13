"""
Forward surrogate network: a neural network that LEARNS the waveguide physics.

This is the machine-learning core of the modern inverse-design framework
(Peurifoy et al., Sci. Adv. 2018; Liu et al., ACS Photonics 2018; and the
2024-2025 AR-waveguide tandem papers in CITATIONS.md Group C): a deep network
is trained on simulated (design -> performance) pairs until it can emulate the
physics itself. Every later stage — the tandem inverse network and the
neural-adjoint design search — then optimizes THROUGH this trained network.

    design theta (8)  ──►  ForwardNet (MLP)  ──►  predicted spec y_hat (4)
    trained on N samples labeled by the analytic physics engine
    (waveguide_physics.py), exactly as the cited works train on RCWA/FDTD data.

The exact physics stays available as ground truth, so every surrogate
prediction can be honestly scored (parity plots, R^2 per metric) — that
physics-anchored verification is this project's twist on the standard recipe.

Usage:
    python3 surrogate.py --pmma --quick     # ~1 min smoke test
    python3 surrogate.py --pmma             # standard run
    python3 surrogate.py --pmma --samples 100000 --epochs 150 --seed 0
Outputs:
    forward_surrogate_seed{N}.pt   per-seed checkpoint
    forward_surrogate.pt           best checkpoint across all runs (auto-kept)
    surrogate_loss_curve.png       training/validation curves
    surrogate_parity.png           predicted-vs-true parity, R^2 per metric
    surrogate_runs.csv             one line per run, for the paper's table
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn

from waveguide_physics import (
    BOUNDS, SPEC_SCALE, forward_model, sample_theta, normalize_theta,
    normalize_spec, use_pmma,
)

SPEC_NAMES = ["MTF@40cyc/mm", "Transmission", "ChromSpread(deg)", "T@FOV"]

# figure palette (validated colorblind-safe set, fixed assignment everywhere)
C_TRAIN, C_VAL, C_SURR, C_REF = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
C_IDENT = "#8a8f98"


class ForwardNet(nn.Module):
    """MLP: normalized design z in [0,1]^8 -> normalized spec (4)."""

    def __init__(self, hidden: int = 256, depth: int = 4):
        super().__init__()
        layers, d_in = [], 8
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.SiLU()]
            d_in = hidden
        layers += [nn.Linear(hidden, 4)]
        self.net = nn.Sequential(*layers)
        self.hidden, self.depth = hidden, depth

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def make_dataset(n: int, seed: int):
    """(normalized design, normalized true spec) pairs labeled by exact physics."""
    g = torch.Generator().manual_seed(seed)
    theta = sample_theta(n, generator=g)
    with torch.no_grad():
        y_n = normalize_spec(forward_model(theta))
    return normalize_theta(theta), y_n


def load_surrogate(path: str = "forward_surrogate.pt") -> ForwardNet:
    """Load a trained surrogate AND restore the design bounds it was trained
    with (PMMA vs full space), so input normalization matches training."""
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found — train the forward surrogate first:\n"
            f"    python3 surrogate.py --pmma")
    ck = torch.load(path, weights_only=True)
    BOUNDS.copy_(ck["bounds"])
    net = ForwardNet(hidden=ck["hidden"], depth=ck["depth"])
    net.load_state_dict(ck["model"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    mode = "PMMA-only" if ck.get("pmma") else "full material"
    print(f"[surrogate] loaded {path} ({mode} space, "
          f"val MSE {ck['best_val']:.6f}, seed {ck['seed']})")
    return net


def r2_per_metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    ss_res = ((y_pred - y_true) ** 2).sum(dim=0)
    ss_tot = ((y_true - y_true.mean(dim=0)) ** 2).sum(dim=0)
    return 1.0 - ss_res / ss_tot


def train(n_train=50000, epochs=80, batch=512, lr=1e-3, seed=0, quick=False,
          pmma=False):
    if quick:
        n_train, epochs = 8000, 15
    n_val = n_test = max(n_train // 10, 500)

    steps_per_epoch = (n_train + batch - 1) // batch
    print(f"config: {n_train} samples | {epochs} epochs | batch {batch} | "
          f"lr {lr} | seed {seed} -> {steps_per_epoch * epochs:,} iterations")

    torch.manual_seed(seed)
    z_tr, y_tr = make_dataset(n_train, seed=100 + seed)
    z_va, y_va = make_dataset(n_val, seed=200)
    z_te, y_te = make_dataset(n_test, seed=300)

    # inverse-spread loss weights: in PMMA mode some metrics (e.g. MTF) span
    # a narrow range; without this the network ignores them and their R^2 dies.
    # 1/std (not 1/var) plus a clamp so no metric starves the others.
    w_loss = 1.0 / (y_tr.std(dim=0) + 1e-6)
    w_loss = (w_loss / w_loss.mean()).clamp(0.25, 4.0)
    print("per-metric loss weights:",
          [f"{n}={w:.2f}" for n, w in zip(SPEC_NAMES, w_loss.tolist())])

    model = ForwardNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t0 = time.time()
    history, best_va, best_state = [], float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        tot = 0.0
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            loss = (w_loss * (model(z_tr[idx]) - y_tr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            va = (w_loss * (model(z_va) - y_va) ** 2).mean().item()
        history.append((ep, tot / n_train, va))
        if va < best_va:
            best_va = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:3d}/{epochs} | train MSE {tot/n_train:.6f} | "
              f"val MSE {va:.6f} | {time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    model.eval()

    # held-out test report: can the network really emulate the physics?
    with torch.no_grad():
        y_hat = model(z_te)
    r2 = r2_per_metric(y_hat, y_te)
    mae_phys = (y_hat - y_te).abs().mean(dim=0) * SPEC_SCALE  # physical units
    print(f"\n=== Surrogate accuracy on {n_test} unseen designs ===")
    for nm, r, a in zip(SPEC_NAMES, r2, mae_phys):
        print(f"  {nm:18s} R^2 {r:.5f}   MAE {a:.5f} (physical units)")
    print(f"  best validation MSE: {best_va:.6f}")

    ckpt = {"model": model.state_dict(), "hidden": model.hidden,
            "depth": model.depth, "bounds": BOUNDS.clone(), "pmma": pmma,
            "best_val": best_va, "seed": seed,
            "r2": r2.tolist(), "n_train": n_train, "epochs": epochs}
    torch.save(ckpt, f"forward_surrogate_seed{seed}.pt")
    print(f"Saved weights -> forward_surrogate_seed{seed}.pt")

    # keep forward_surrogate.pt = the best surrogate ever trained in this space
    keep_old = False
    if os.path.exists("forward_surrogate.pt"):
        old = torch.load("forward_surrogate.pt", weights_only=True)
        keep_old = old.get("pmma") == pmma and old["best_val"] <= best_va
    if keep_old:
        print("forward_surrogate.pt kept (existing checkpoint is better)")
    else:
        torch.save(ckpt, "forward_surrogate.pt")
        print("forward_surrogate.pt updated (new best for this design space)")

    new = not os.path.exists("surrogate_runs.csv")
    with open("surrogate_runs.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["seed", "pmma", "samples", "epochs", "batch", "lr",
                        "iterations", "best_val_MSE"] +
                       [f"R2_{n}" for n in SPEC_NAMES])
        w.writerow([seed, int(pmma), n_train, epochs, batch, lr,
                    steps_per_epoch * epochs, f"{best_va:.6f}"] +
                   [f"{v:.5f}" for v in r2.tolist()])
    print("Appended run summary -> surrogate_runs.csv")

    make_figures(history, y_hat, y_te, r2)
    return model


def make_figures(history, y_hat, y_te, r2):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to also get the surrogate figures)")
        return

    eps, tr, va = zip(*history)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(eps, tr, color=C_TRAIN, label="train MSE")
    ax.semilogy(eps, va, color=C_VAL, label="validation MSE")
    ax.set_xlabel("epoch (pass through dataset)")
    ax.set_ylabel("loss (log scale, lower = better)")
    ax.set_title("Forward surrogate network — learning the physics")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig("surrogate_loss_curve.png", dpi=150)
    plt.close(fig)
    print("Saved training curve -> surrogate_loss_curve.png")

    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
    for k, ax in enumerate(axes.flat):
        t, p = y_te[:, k].numpy(), y_hat[:, k].numpy()
        lo, hi = float(min(t.min(), p.min())), float(max(t.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "--", color=C_IDENT, lw=1, zorder=1)
        ax.scatter(t, p, s=6, alpha=0.35, color=C_TRAIN, edgecolors="none",
                   zorder=2)
        ax.set_title(f"{SPEC_NAMES[k]}   R$^2$={r2[k]:.4f}", fontsize=10)
        ax.set_xlabel("true (exact physics)"); ax.set_ylabel("network prediction")
        ax.grid(alpha=0.25)
    fig.suptitle("Surrogate parity on unseen designs (normalized units)")
    fig.tight_layout(); fig.savefig("surrogate_parity.png", dpi=150)
    plt.close(fig)
    print("Saved parity plot -> surrogate_parity.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="small fast run for smoke test")
    p.add_argument("--pmma", action="store_true", help="PMMA-only: pin material, design geometry")
    p.add_argument("--samples", type=int, default=50000, help="training set size")
    p.add_argument("--epochs", type=int, default=80, help="passes through the dataset")
    p.add_argument("--batch", type=int, default=512, help="batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--seed", type=int, default=0, help="random seed (run >=5 seeds for paper)")
    args = p.parse_args()
    if args.pmma:
        use_pmma()
    train(n_train=args.samples, epochs=args.epochs, batch=args.batch,
          lr=args.lr, seed=args.seed, quick=args.quick, pmma=args.pmma)
