"""
Physics-anchored tandem inverse design of AR waveguides.

Inverse network g: target spec y* (4) -> design theta_hat (8, via sigmoid -> bounds).
Loss is computed in SPEC space through the frozen differentiable physics engine:
    L = || f(g(y*)) - y* ||^2  (normalized per-metric)
This sidesteps design non-uniqueness (one-to-many y->theta), the classic tandem trick,
but with EXACT physics as the decoder instead of a learned surrogate.

Also trains a naive direct-regression baseline (y -> theta with theta-space MSE)
to demonstrate the non-uniqueness failure mode for the paper's baseline table.

Usage:  python3 train_inverse.py [--quick]
"""

import argparse
import time

import torch
import torch.nn as nn

from waveguide_physics import (
    forward_model, sample_theta, denormalize_theta, normalize_theta,
    normalize_spec,
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


def train(n_train=30000, n_val=3000, epochs=40, batch=512, lr=1e-3, quick=False):
    if quick:
        n_train, n_val, epochs = 6000, 1000, 8

    torch.manual_seed(0)
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
    for ep in range(epochs):
        perm = torch.randperm(n_train)
        tot, tot_b = 0.0, 0.0
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            yb = y_tr_n[idx]

            # --- tandem: spec-space loss through frozen physics
            z_hat = model(yb)
            theta_hat = denormalize_theta(z_hat)
            y_hat = normalize_spec(forward_model(theta_hat))
            loss = nn.functional.mse_loss(y_hat, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)

            # --- baseline: theta-space loss (no physics in the loop)
            loss_b = nn.functional.mse_loss(baseline(yb), z_tr[idx])
            opt_b.zero_grad(); loss_b.backward(); opt_b.step()
            tot_b += loss_b.item() * len(idx)
        sched.step()

        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            va = evaluate(model, y_va, y_va_n, quiet=True)
            print(f"ep {ep:3d} | tandem train {tot/n_train:.5f} | "
                  f"val spec-MSE {va:.5f} | baseline theta-MSE {tot_b/n_train:.5f} | "
                  f"{time.time()-t0:.0f}s")

    print("\n=== Held-out evaluation (tandem, spec-space) ===")
    evaluate(model, y_va, y_va_n)
    print("\n=== Held-out evaluation (naive baseline, spec-space) ===")
    evaluate(baseline, y_va, y_va_n)

    torch.save({"model": model.state_dict(), "baseline": baseline.state_dict()},
               "inverse_model.pt")
    print("\nSaved weights -> inverse_model.pt")
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
    args = p.parse_args()
    m = train(quick=args.quick)
    demo_reverse_engineering(m)
