"""
Direct gradient optimization of PMMA waveguide designs.

Different job from train_inverse.py:
  train_inverse.py : "here is a target spec -> give me a design that matches it"
  optimize_pmma.py : "search the whole PMMA design space -> give me the BEST designs"

Because the physics engine is differentiable, we don't need a neural network for
this — we run gradient ASCENT on the design parameters themselves, from many
random starting points (multi-start avoids local optima), maximizing a weighted
objective:

    J = w_mtf * MTF  +  w_T * (T / 0.10)  -  w_ca * (chrom_spread / 30 deg)

Weights encode the design priorities; sweep them to trace the trade-off frontier
(the Pareto front) — that trade-off curve is a headline figure for the paper.

Usage:  python3 optimize_pmma.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import res_path
from physics.waveguide_physics import (
    forward_model, use_pmma, sample_theta, normalize_theta, denormalize_theta,
    tir_penalty, transmission_polarized, fov_window_deg,
)

N_STARTS = 300     # random starting designs
N_STEPS = 400      # gradient steps
LR = 0.03
W_MTF, W_T, W_CA = 1.0, 1.0, 0.5   # objective weights — sweep these for Pareto front

LABELS = ["n", "alpha(1/mm)", "sigma(nm)", "Lc(nm)", "t(mm)",
          "period(nm)", "depth(nm)", "duty"]


W_TIR = 10.0   # weight of the TIR-feasibility penalty (FIX-1); designs whose
               # RGB orders leave the guiding window are steered back in


def objective(theta: torch.Tensor) -> torch.Tensor:
    y = forward_model(theta)                       # [B,4]
    mtf, T, ca, T_fov = y.unbind(dim=1)
    return (W_MTF * mtf + W_T * (T / 0.10) + 0.3 * (T_fov / 0.10)
            - W_CA * (ca / 30.0) - W_TIR * tir_penalty(theta))


def optimize():
    use_pmma()
    torch.manual_seed(0)

    # optimize in unconstrained logit space; sigmoid keeps designs inside bounds
    z0 = normalize_theta(sample_theta(N_STARTS)).clamp(1e-3, 1 - 1e-3)
    w = torch.log(z0 / (1 - z0))                   # logit
    w.requires_grad_(True)
    opt = torch.optim.Adam([w], lr=LR)

    for step in range(N_STEPS):
        theta = denormalize_theta(torch.sigmoid(w))
        loss = -objective(theta).sum()             # ascent via negative descent
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"step {step:4d} | best J so far: {objective(theta).max():.4f}")

    with torch.no_grad():
        theta = denormalize_theta(torch.sigmoid(w))
        J = objective(theta)
        y = forward_model(theta)
        top = torch.topk(J, k=5).indices

    pol = transmission_polarized(theta)
    fov_lo, fov_hi, fov_w = fov_window_deg(theta)

    print("\n=== Top 5 optimal PMMA designs ===")
    for rank, i in enumerate(top, 1):
        mtf, T, ca, T_fov = y[i].tolist()
        print(f"\n#{rank}  J={J[i]:.4f} | MTF {mtf:.4f} | T {100*T:.2f}% | "
              f"chrom {ca:.2f} deg | T@FOV {100*T_fov:.2f}%")
        print(f"    T_TE {100*pol['TE'][i]:.2f}%  T_TM {100*pol['TM'][i]:.2f}%  "
              f"diattenuation {pol['diattenuation'][i]:.3f}")
        print(f"    full-RGB guided FOV window: [{fov_lo[i]:.1f}, {fov_hi[i]:.1f}] deg "
              f"(width {fov_w[i]:.1f} deg)")
        for lb, v in zip(LABELS, theta[i].tolist()):
            print(f"    {lb:12s} = {v:,.4g}")

    # permanent record of the winners (now with polarization + FOV window)
    import csv
    with open(res_path("optimal_designs.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["rank", "J", "MTF", "T", "chrom_deg", "T_fov",
                       "T_TE", "T_TM", "fov_lo_deg", "fov_hi_deg"] + LABELS)
        for rank, i in enumerate(top, 1):
            wcsv.writerow([rank, f"{J[i]:.4f}"] +
                          [f"{v:.5g}" for v in y[i].tolist()] +
                          [f"{pol['TE'][i]:.5g}", f"{pol['TM'][i]:.5g}",
                           f"{fov_lo[i]:.3f}", f"{fov_hi[i]:.3f}"] +
                          [f"{v:.5g}" for v in theta[i].tolist()])
    print("\nSaved winners -> results/optimal_designs.csv")

    # hall of fame: keep the best design EVER seen across all runs; only
    # updates when a new run beats the record
    hof = res_path("best_design_ever_v2.csv")   # v2: records from the corrected (TIR-
    # constrained, polarization-resolved) physics; v1 records are not comparable
    prev_J = -1e9
    if os.path.exists(hof):
        with open(hof) as f:
            try:
                prev_J = float(list(csv.reader(f))[1][1])
            except (IndexError, ValueError):
                pass
    i_best = top[0]
    if J[i_best] > prev_J:
        with open(hof, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["date", "J", "MTF", "T", "chrom_deg", "T_fov"] + LABELS)
            import datetime
            wcsv.writerow([datetime.date.today().isoformat(), f"{J[i_best]:.8f}"] +
                          [f"{v:.5g}" for v in y[i_best].tolist()] +
                          [f"{v:.5g}" for v in theta[i_best].tolist()])
        print(f"NEW RECORD: J={J[i_best]:.4f} beats {prev_J:.4f} -> {hof}")
    else:
        print(f"no new record (best this run {J[i_best]:.4f} vs all-time {prev_J:.4f})")

    # diversity check: are the optima one basin or several?
    geo = normalize_theta(theta[top])[:, 4:]       # geometry dims only
    spread = geo.std(dim=0).mean().item()
    print(f"\ngeometry diversity across top-5 (normalized std): {spread:.3f}"
          f"  ({'multiple distinct optima' if spread > 0.05 else 'single basin'})")


if __name__ == "__main__":
    optimize()
