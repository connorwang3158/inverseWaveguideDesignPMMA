"""
Tandem inverse design of AR waveguides — neural network trained through a
neural network.

Inverse network g: target spec y* (4) -> design theta_hat (8, via sigmoid -> bounds).
The loss is computed in SPEC space through a frozen DECODER:

  --decoder surrogate (default)
      the trained forward surrogate network from surrogate.py — the classic
      tandem architecture of modern inverse-design frameworks (Liu et al.,
      ACS Photonics 2018; Optics Express 32, 12587, 2024): both halves of the
      pipeline are neural networks, and the inverse net learns by
      back-propagating through the learned physics emulator.
  --decoder physics
      the exact differentiable analytic engine — the physics-anchored
      ablation that quantifies how much error the surrogate contributes.

    L = || decoder(g(y*)) - y* ||^2   (normalized per-metric)

Spec-space loss sidesteps design non-uniqueness (one-to-many y->theta), the
classic tandem trick. Validation is ALWAYS scored against the exact physics,
whichever decoder trains the network — honest numbers only.

Also trains a naive direct-regression baseline (y -> theta with theta-space MSE)
to demonstrate the non-uniqueness failure mode for the paper's baseline table.

Usage:
    python3 surrogate.py --pmma            # first: train the forward surrogate
    python3 train_inverse.py --pmma        # then: tandem through the surrogate
    python3 train_inverse.py --pmma --decoder physics   # ablation
"""

import argparse
import time

import torch
import torch.nn as nn

from waveguide_physics import (
    forward_model, sample_theta, denormalize_theta, normalize_theta,
    normalize_spec, use_pmma,
)

C_TRAIN, C_VAL, C_SURR = "#2a78d6", "#1baf7a", "#eda100"  # shared figure palette


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


def physics_decoder(z_hat: torch.Tensor) -> torch.Tensor:
    """Normalized design -> normalized spec via the exact analytic engine."""
    return normalize_spec(forward_model(denormalize_theta(z_hat)))


def train(n_train=30000, n_val=3000, epochs=40, batch=512, lr=1e-3, quick=False,
          seed=0, decoder="surrogate"):
    if quick:
        n_train, n_val, epochs = 6000, 1000, 8

    # load the frozen decoder FIRST — loading the surrogate also restores the
    # design bounds it was trained with, so the dataset matches its space
    if decoder == "surrogate":
        from surrogate import load_surrogate
        surr = load_surrogate()
        dec = surr
    else:
        surr, dec = None, physics_decoder

    steps_per_epoch = (n_train + batch - 1) // batch
    print(f"config: {n_train} samples | {epochs} epochs | batch {batch} | "
          f"lr {lr} | seed {seed} | decoder {decoder} -> "
          f"{steps_per_epoch * epochs:,} iterations")

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
            y_hat = dec(z_hat)
            loss = nn.functional.mse_loss(y_hat, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)

            # --- baseline: theta-space loss (no decoder in the loop)
            loss_b = nn.functional.mse_loss(baseline(yb), z_tr[idx])
            opt_b.zero_grad(); loss_b.backward(); opt_b.step()
            tot_b += loss_b.item() * len(idx)
        sched.step()

        # every epoch: score on held-out data. Physics val is the number that
        # counts; surrogate val shows what the network itself believes.
        va = evaluate(model, y_va, y_va_n, quiet=True)
        with torch.no_grad():
            va_surr = (nn.functional.mse_loss(dec(model(y_va_n)), y_va_n).item()
                       if surr is not None else float("nan"))
        history.append((ep, tot / n_train, va, va_surr))
        if va < best_va:  # checkpoint the best model seen, not just the last
            best_va = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        surr_txt = f" | surr-val {va_surr:.5f}" if surr is not None else ""
        print(f"ep {ep:3d}/{epochs} | train loss {tot/n_train:.5f} | "
              f"val spec-MSE {va:.5f}{surr_txt} | "
              f"baseline theta-MSE {tot_b/n_train:.5f} | {time.time()-t0:.0f}s")

    print("\n=== Held-out evaluation vs EXACT physics (tandem) ===")
    evaluate(model, y_va, y_va_n)
    print("\n=== Held-out evaluation vs EXACT physics (naive baseline) ===")
    evaluate(baseline, y_va, y_va_n)

    if best_state is not None:
        model.load_state_dict(best_state)  # report/save the best epoch, not the last
    print(f"\nbest validation spec-MSE (exact physics): {best_va:.6f}")
    torch.save({"model": model.state_dict(), "baseline": baseline.state_dict(),
                "best_val": best_va, "seed": seed, "decoder": decoder},
               f"inverse_model_seed{seed}.pt")
    print(f"Saved weights -> inverse_model_seed{seed}.pt")

    # permanent record: one line per training run, for the paper's 5-seed table
    import csv, os
    new = not os.path.exists("training_runs.csv")
    with open("training_runs.csv", "a", newline="") as f:
        wcsv = csv.writer(f)
        if new:
            wcsv.writerow(["seed", "decoder", "samples", "epochs", "batch", "lr",
                           "iterations", "best_val_specMSE"])
        wcsv.writerow([seed, decoder, n_train, epochs, batch, lr,
                       steps_per_epoch * epochs, f"{best_va:.6f}"])
    print("Appended run summary -> training_runs.csv")

    try:  # loss curve figure — visual record of the training run
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eps, tr, va_p, va_s = zip(*history)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(eps, tr, color=C_TRAIN, label="train loss")
        ax.semilogy(eps, va_p, color=C_VAL, label="validation (exact physics)")
        if surr is not None:
            ax.semilogy(eps, va_s, color=C_SURR, ls="--",
                        label="validation (surrogate's own belief)")
        ax.set_xlabel("epoch (pass through dataset)")
        ax.set_ylabel("loss (log scale, lower = better)")
        ax.set_title(f"Inverse network training (decoder: {decoder})")
        ax.grid(alpha=0.25); ax.legend()
        fig.tight_layout(); fig.savefig("loss_curve.png", dpi=150)
        print("Saved training curve -> loss_curve.png")
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
    names = ["MTF@40cyc/mm", "Transmission", "ChromSpread(deg)", "T@FOV"]
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
    # target: sharp (MTF 0.65), efficient-for-class (T 6.5%), low chrom spread
    y_star = torch.tensor([[0.65, 0.065, 14.0, 0.045]])
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
    p.add_argument("--decoder", choices=["surrogate", "physics"], default="surrogate",
                   help="frozen decoder the tandem trains through (default: the "
                        "learned surrogate network; 'physics' = exact-engine ablation)")
    p.add_argument("--samples", type=int, default=30000, help="training set size")
    p.add_argument("--epochs", type=int, default=40, help="passes through the dataset")
    p.add_argument("--batch", type=int, default=512, help="batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--seed", type=int, default=0, help="random seed (run >=5 seeds for paper)")
    args = p.parse_args()
    if args.pmma:
        use_pmma()
    m = train(n_train=args.samples, epochs=args.epochs, batch=args.batch,
              lr=args.lr, quick=args.quick, seed=args.seed, decoder=args.decoder)
    demo_reverse_engineering(m)
