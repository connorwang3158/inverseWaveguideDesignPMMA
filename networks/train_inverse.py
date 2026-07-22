"""
Tandem inverse design of AR waveguides (Liu et al., ACS Photonics 2018).

Inverse network g: target spec y* (4) -> design theta_hat (8, via sigmoid -> bounds).
Loss is computed in SPEC space through a frozen decoder f:
    L = || f(g(y*)) - y* ||^2  (normalized per-metric)
This sidesteps design non-uniqueness (one-to-many y->theta), the classic tandem trick.

Two decoders, selected with --decoder:
    surrogate  (default)  the TRAINED forward network from surrogate.py, the
                          standard tandem recipe of the cited literature; run
                          `python3 surrogate.py --pmma` first
    physics               the exact differentiable physics engine, this
                          project's ablation arm for the paper's comparison
Validation is ALWAYS scored by the exact physics, whichever decoder trains.

Also trains a naive direct-regression baseline (y -> theta with theta-space MSE)
to demonstrate the non-uniqueness failure mode for the paper's baseline table.

Usage:  python3 train_inverse.py --pmma [--decoder surrogate|physics] [--quick]
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ckpt_path, fig_path, res_path
from physics.waveguide_physics import (
    ENGINE_VERSION, forward_model, sample_theta, denormalize_theta,
    normalize_theta, normalize_spec, use_pmma,
)


class InverseNet(nn.Module):
    """MLP: normalized spec (4) -> normalized design z in (0,1)^8."""

    def __init__(self, hidden: int = 256, depth: int = 4):
        super().__init__()
        layers, d_in = [], 4
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.SiLU()]
            d_in = hidden
        layers += [nn.Linear(hidden, 8), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, y_norm: torch.Tensor) -> torch.Tensor:
        return self.net(y_norm)


def make_dataset(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    theta = sample_theta(n, generator=g)
    with torch.no_grad():
        y = forward_model(theta)
    return theta, y


def train(n_train=30000, n_val=3000, epochs=40, batch=512, lr=1e-3, quick=False,
          seed=0, decoder="surrogate"):
    if quick:
        n_train, n_val, epochs = 6000, 1000, 8

    # frozen decoder for the tandem loss. The surrogate loader also restores
    # the design space it was trained in, so build it BEFORE the datasets.
    if decoder == "surrogate":
        from surrogate import load_surrogate
        surr = load_surrogate()

        def decode(z_hat):           # both nets live in normalized spaces
            return surr(z_hat)
        print("decoder: frozen trained surrogate (tandem recipe of the cited papers)")
    else:
        def decode(z_hat):
            return normalize_spec(forward_model(denormalize_theta(z_hat)))
        print("decoder: exact differentiable physics (ablation arm)")

    steps_per_epoch = (n_train + batch - 1) // batch
    print(f"config: {n_train} samples | {epochs} epochs | batch {batch} | "
          f"lr {lr} | seed {seed} -> {steps_per_epoch * epochs:,} iterations")

    torch.manual_seed(seed)
    theta_tr, y_tr = make_dataset(n_train, seed=1)
    theta_va, y_va = make_dataset(n_val, seed=2)
    y_tr_n, y_va_n = normalize_spec(y_tr), normalize_spec(y_va)

    model = InverseNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # naive baseline: direct theta-regression (shows non-uniqueness failure)
    baseline = InverseNet()
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=lr)
    z_tr = normalize_theta(theta_tr)

    t0 = time.time()
    history = []
    best_va, best_state = float("inf"), None
    for ep in range(epochs):
        perm = torch.randperm(n_train)
        tot, tot_b = 0.0, 0.0
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            yb = y_tr_n[idx]

            # --- tandem: spec-space loss through the frozen decoder
            z_hat = model(yb)
            y_hat = decode(z_hat)
            loss = nn.functional.mse_loss(y_hat, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)

            # --- baseline: theta-space loss (no physics in the loop)
            loss_b = nn.functional.mse_loss(baseline(yb), z_tr[idx])
            opt_b.zero_grad(); loss_b.backward(); opt_b.step()
            tot_b += loss_b.item() * len(idx)
        sched.step()

        # every epoch: evaluate on held-out data and log, so progress is watchable
        va = evaluate(model, y_va, y_va_n, quiet=True)
        history.append((ep, tot / n_train, va))
        if va < best_va:  # checkpoint the best model seen, not just the last
            best_va = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:3d}/{epochs} | train loss {tot/n_train:.5f} | "
              f"val spec-MSE {va:.5f} | baseline theta-MSE {tot_b/n_train:.5f} | "
              f"{time.time()-t0:.0f}s")

    if best_state is not None:
        model.load_state_dict(best_state)  # report/save the best epoch, not the
        # last, restored BEFORE the final evaluation so the printed table and
        # the saved checkpoint describe the SAME model (previously the printout
        # scored the last epoch while the checkpoint stored the best epoch)

    print("\n=== Held-out evaluation (tandem, spec-space, best epoch) ===")
    evaluate(model, y_va, y_va_n)
    print("\n=== Held-out evaluation (naive baseline, spec-space) ===")
    evaluate(baseline, y_va, y_va_n)
    print(f"\nbest validation spec-MSE (exact physics): {best_va:.6f}")
    torch.save({"model": model.state_dict(), "baseline": baseline.state_dict(),
                "best_val": best_va, "seed": seed, "decoder": decoder},
               ckpt_path(f"inverse_model_seed{seed}.pt"))
    print(f"Saved weights -> checkpoints/inverse_model_seed{seed}.pt")

    # permanent record: one line per training run, for the paper's 5-seed table
    import csv
    # keyed on the physics engine version, like surrogate_runs (v2-era rows
    # live in training_runs.csv and are not comparable across engines)
    runs_csv = res_path(f"training_runs_{ENGINE_VERSION}.csv")
    new = not os.path.exists(runs_csv)
    with open(runs_csv, "a", newline="") as f:
        wcsv = csv.writer(f)
        if new:
            wcsv.writerow(["seed", "decoder", "samples", "epochs", "batch",
                           "lr", "iterations", "best_val_specMSE"])
        wcsv.writerow([seed, decoder, n_train, epochs, batch, lr,
                       steps_per_epoch * epochs, f"{best_va:.6f}"])
    print(f"Appended run summary -> {runs_csv}")

    try:  # loss curve figure, visual record of the training run
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eps, tr, va = zip(*history)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(eps, tr, label="train loss")
        ax.semilogy(eps, va, label="validation spec-MSE")
        ax.set_xlabel("epoch (pass through dataset)")
        ax.set_ylabel("loss (log scale, lower = better)")
        ax.set_title("Inverse network training progress")
        ax.legend(); fig.tight_layout()
        fig.savefig(fig_path("loss_curve.png"), dpi=150)
        print("Saved training curve -> figures/loss_curve.png")
    except ImportError:
        print("(install matplotlib to also get loss_curve.png)")
    return model


@torch.no_grad()
def evaluate(model, y_true, y_true_n, quiet=False):
    theta_hat = denormalize_theta(model(y_true_n))
    y_hat = forward_model(theta_hat)
    mse = nn.functional.mse_loss(normalize_spec(y_hat), y_true_n).item()
    if quiet:
        return mse
    names = ["MTF@40cyc/mm", "T_FOM", "PupilWalkoff(mm)", "T@FOV"]
    mae = (y_hat - y_true).abs().mean(dim=0)
    rel = ((y_hat - y_true).abs() / (y_true.abs() + 1e-8)).median(dim=0).values
    for nm, a, r in zip(names, mae, rel):
        print(f"  {nm:18s} MAE {a:.5f}   median rel err {100*r:.2f}%")
    print(f"  normalized spec MSE: {mse:.6f}")
    return mse


@torch.no_grad()
def demo_reverse_engineering(model):
    """Headline demo: request a target spec, read back the recovered design."""
    print("\n=== Reverse-engineering demo ===")
    # target: sharp-for-class (MTF 0.45 under the v5 Watson-eye anchor; the
    # gradient-probed ceiling of the v5 PMMA space is ~0.466), efficient-
    # for-class (T_FOM 1.1% vs ~1.35% ceiling), mid-range pupil walk-off
    # (floor ~1.52 mm in the guided window). NOTE: the target must be
    # physically reachable, an unreachable request makes the demo look
    # like a network failure rather than an infeasible spec (lesson from
    # the v3-era 14-deg target).
    y_star = torch.tensor([[0.45, 0.011, 1.7, 0.010]])
    theta = denormalize_theta(model(normalize_spec(y_star)))
    y_ach = forward_model(theta)
    labels = ["n", "alpha(1/mm)", "sigma(nm)", "Lc(nm)", "t(mm)",
              "period(nm)", "depth(nm)", "duty"]
    print("target spec   :", [round(v, 4) for v in y_star[0].tolist()])
    print("achieved spec :", [round(v, 4) for v in y_ach[0].tolist()])
    print("recovered design:")
    for lb, v in zip(labels, theta[0].tolist()):
        print(f"  {lb:12s} = {v:,.4g}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="small fast run for smoke test")
    p.add_argument("--pmma", action="store_true", help="PMMA-only: pin material, design geometry")
    p.add_argument("--samples", type=int, default=30000, help="training set size")
    p.add_argument("--epochs", type=int, default=40, help="passes through the dataset")
    p.add_argument("--batch", type=int, default=512, help="batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--seed", type=int, default=0, help="random seed (run >=5 seeds for paper)")
    p.add_argument("--decoder", choices=["surrogate", "physics"],
                   default="surrogate",
                   help="tandem decoder: trained surrogate network (default) "
                        "or exact physics (ablation)")
    args = p.parse_args()
    if args.pmma:
        use_pmma()
    m = train(n_train=args.samples, epochs=args.epochs, batch=args.batch,
              lr=args.lr, quick=args.quick, seed=args.seed,
              decoder=args.decoder)
    demo_reverse_engineering(m)
