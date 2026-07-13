"""
Neural-adjoint design search: optimize waveguide designs THROUGH the trained
forward surrogate network.

This is the neural-adjoint method (Ren, Padilla & Malof, NeurIPS 2020): freeze
the trained ForwardNet, treat the DESIGN PARAMETERS as the trainable variables,
and run gradient ascent where every gradient is back-propagated through the
neural network's weights — the network's learned understanding of the physics
steers the search. Multi-start (hundreds of random initial designs in parallel)
avoids local optima.

    theta (trainable) ──► frozen ForwardNet ──► predicted spec ──► objective J
        ▲                                                             │
        └───────────── d J / d theta  (through the network) ◄────────┘

    J = w_mtf * MTF + w_T * (T / 0.10) + 0.3 * (T_fov / 0.10) - w_ca * (chrom / 30)

Honesty step: every finalist design is re-scored with the EXACT physics engine,
ranked by its TRUE objective, and the surrogate-vs-physics gap is reported and
plotted — a design only counts if the real physics agrees with the network.

Usage (from the project root):
    python3 networks/surrogate.py --pmma        # first: train the surrogate
    python3 networks/neural_adjoint.py          # then: search through it
    python3 networks/neural_adjoint.py --quick  # fast smoke test
Outputs:
    results/optimal_designs_na.csv    top designs (surrogate-found, physics-verified)
    figures/neural_adjoint_run.png    search trajectory + surrogate-vs-physics parity
    results/best_design_ever_v2.csv   all-time hall of fame (physics-scored, shared
                                 with optimize_pmma.py; v2 = records under the
                                 corrected TIR-constrained physics)
"""

import argparse
import csv
import datetime
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import fig_path, res_path
from physics.waveguide_physics import (
    SPEC_SCALE, forward_model, sample_theta, normalize_theta,
    denormalize_theta, tir_penalty,
)
from networks.surrogate import load_surrogate

W_MTF, W_T, W_CA = 1.0, 1.0, 0.5   # objective weights (match optimize_pmma.py)
W_TIR = 10.0   # TIR-feasibility penalty weight (FIX-1) — same as optimize_pmma,
               # so the two methods maximize the SAME objective and their hall-
               # of-fame scores are directly comparable

LABELS = ["n", "alpha(1/mm)", "sigma(nm)", "Lc(nm)", "t(mm)",
          "period(nm)", "depth(nm)", "duty"]
SPEC_NAMES = ["MTF", "T", "chrom(deg)", "T@FOV"]
C_TRAIN, C_VAL, C_SURR, C_REF = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
C_VIOLET, C_IDENT = "#4a3aa7", "#8a8f98"


def objective_from_spec(y: torch.Tensor) -> torch.Tensor:
    """y [B,4] in PHYSICAL units -> scalar objective per design."""
    mtf, T, ca, T_fov = y.unbind(dim=1)
    return W_MTF * mtf + W_T * (T / 0.10) + 0.3 * (T_fov / 0.10) - W_CA * (ca / 30.0)


def search(n_starts=400, n_steps=600, lr=0.05, topk=5, seed=0, quick=False):
    if quick:
        n_starts, n_steps = 64, 120

    surr = load_surrogate()          # frozen network; also restores its bounds
    torch.manual_seed(seed)

    # optimize in unconstrained logit space; sigmoid keeps designs inside bounds
    z0 = normalize_theta(sample_theta(n_starts)).clamp(1e-3, 1 - 1e-3)
    w = torch.log(z0 / (1 - z0))
    w.requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)

    traj = []                        # best surrogate-predicted J per step
    for step in range(n_steps):
        z = torch.sigmoid(w)
        y_pred = surr(z) * SPEC_SCALE          # network prediction, physical units
        # TIR penalty comes from the design itself (not the surrogate), keeping
        # the search inside the guiding window exactly like optimize_pmma.py
        J = objective_from_spec(y_pred) - W_TIR * tir_penalty(denormalize_theta(z))
        opt.zero_grad(); (-J.sum()).backward(); opt.step()
        traj.append(J.max().item())
        if step % 100 == 0:
            print(f"step {step:4d} | best J (network's belief): {traj[-1]:.4f}")

    # --- verification: re-score EVERY finalist with the exact physics engine
    with torch.no_grad():
        z = torch.sigmoid(w)
        theta = denormalize_theta(z)
        y_surr = surr(z) * SPEC_SCALE
        y_phys = forward_model(theta)
        pen = W_TIR * tir_penalty(theta)
        J_surr = objective_from_spec(y_surr) - pen
        J_phys = objective_from_spec(y_phys) - pen

    gap = (J_surr - J_phys).abs()
    print(f"\nsurrogate-vs-physics J gap over {n_starts} finalists: "
          f"mean {gap.mean():.4f} | worst {gap.max():.4f}")

    # physical validity flag: all RGB first orders inside the guiding window
    # 1 < sin(theta_i)+lambda/period < n (the v2 engine's FIX-1 inequality).
    # Uses margin=0 — the PHYSICAL window — not the optimizer's 0.01 safety
    # margin, which previously flagged genuinely guided designs near the
    # window edge (e.g. blue at period 449 nm) as unguided (false negatives).
    tir_ok = tir_penalty(theta, margin=0.0) <= 0.0

    top = torch.topk(J_phys, k=topk).indices   # rank by TRUE physics, not belief
    print(f"\n=== Top {topk} designs (found by the network, verified by physics) ===")
    for rank, i in enumerate(top, 1):
        mtf, T, ca, T_fov = y_phys[i].tolist()
        guided = "yes" if tir_ok[i] else "NO (outside TIR guiding window)"
        print(f"\n#{rank}  J={J_phys[i]:.4f} (network believed {J_surr[i]:.4f}) | "
              f"MTF {mtf:.4f} | T {100*T:.2f}% | chrom {ca:.2f} deg | "
              f"T@FOV {100*T_fov:.2f}% | TIR-guided: {guided}")
        for lb, v in zip(LABELS, theta[i].tolist()):
            print(f"    {lb:12s} = {v:,.4g}")

    with open(res_path("optimal_designs_na.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["rank", "J_physics", "J_surrogate",
                       "MTF", "T", "chrom_deg", "T_fov"] + LABELS +
                      ["tir_guided_rgb"])
        for rank, i in enumerate(top, 1):
            wcsv.writerow([rank, f"{J_phys[i]:.4f}", f"{J_surr[i]:.4f}"] +
                          [f"{v:.5g}" for v in y_phys[i].tolist()] +
                          [f"{v:.5g}" for v in theta[i].tolist()] +
                          [int(tir_ok[i])])
    print("\nSaved winners -> results/optimal_designs_na.csv")

    # hall of fame (shared with optimize_pmma.py): physics-scored record only.
    # v2 = records under the corrected (TIR-constrained, polarization-resolved)
    # physics; v1 records were set by leaky designs and are not comparable.
    hof, prev_J = res_path("best_design_ever_v2.csv"), -1e9
    if os.path.exists(hof):
        with open(hof) as f:
            try:
                prev_J = float(list(csv.reader(f))[1][1])
            except (IndexError, ValueError):
                pass
    i_best = top[0]
    if J_phys[i_best] > prev_J:
        with open(hof, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["date", "J", "MTF", "T", "chrom_deg", "T_fov"] + LABELS)
            wcsv.writerow([datetime.date.today().isoformat(),
                           f"{J_phys[i_best]:.8f}"] +
                          [f"{v:.5g}" for v in y_phys[i_best].tolist()] +
                          [f"{v:.5g}" for v in theta[i_best].tolist()])
        print(f"NEW RECORD: J={J_phys[i_best]:.4f} beats {prev_J:.4f} -> {hof}")
    else:
        print(f"no new record (best this run {J_phys[i_best]:.4f} "
              f"vs all-time {prev_J:.4f})")

    make_figure(traj, y_surr, y_phys, top)
    return theta[top], y_phys[top]


def make_figure(traj, y_surr, y_phys, top):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib to also get neural_adjoint_run.png)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(range(len(traj)), traj, color=C_TRAIN, lw=1.5)
    ax1.set_xlabel("gradient step (through the frozen network)")
    ax1.set_ylabel("best predicted objective J")
    ax1.set_title("Neural-adjoint search trajectory")
    ax1.grid(alpha=0.25)

    # parity: does the exact physics agree with the network on the finalists?
    ys, yp = (y_surr / SPEC_SCALE).numpy(), (y_phys / SPEC_SCALE).numpy()
    colors = [C_TRAIN, C_VAL, C_SURR, C_REF]
    lo = float(min(ys.min(), yp.min())); hi = float(max(ys.max(), yp.max()))
    ax2.plot([lo, hi], [lo, hi], "--", color=C_IDENT, lw=1, zorder=1)
    for k, (nm, c) in enumerate(zip(SPEC_NAMES, colors)):
        ax2.scatter(ys[:, k], yp[:, k], s=14, alpha=0.5, color=c,
                    edgecolors="none", label=nm, zorder=2)
    ax2.scatter(ys[top.numpy(), :].flatten(), yp[top.numpy(), :].flatten(),
                s=60, facecolors="none", edgecolors=C_VIOLET, lw=1.2,
                label="top designs", zorder=3)
    ax2.set_xlabel("network prediction (normalized)")
    ax2.set_ylabel("exact physics (normalized)")
    ax2.set_title("Finalists: surrogate vs physics")
    ax2.grid(alpha=0.25); ax2.legend(fontsize=8)

    fig.tight_layout(); fig.savefig(fig_path("neural_adjoint_run.png"), dpi=150)
    plt.close(fig)
    print("Saved figure -> figures/neural_adjoint_run.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="small fast run for smoke test")
    p.add_argument("--starts", type=int, default=400, help="parallel random starting designs")
    p.add_argument("--steps", type=int, default=600, help="gradient steps")
    p.add_argument("--lr", type=float, default=0.05, help="step size")
    p.add_argument("--topk", type=int, default=5, help="designs to report")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    args = p.parse_args()
    search(n_starts=args.starts, n_steps=args.steps, lr=args.lr,
           topk=args.topk, seed=args.seed, quick=args.quick)
