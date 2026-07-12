"""
Trade-off frontier exploration for PMMA waveguide designs.

Sweeps the objective weights in optimize-style gradient ascent to trace how much
MTF you must give up to gain transmission (and vice versa), and how chromatic
spread rides along. Outputs pareto_results.csv and pareto_front.png.

Usage: python3 sweep_pareto.py
"""

import csv

import torch

from waveguide_physics import (
    forward_model, use_pmma, sample_theta, normalize_theta, denormalize_theta,
)

N_STARTS, N_STEPS, LR = 80, 200, 0.03
LABELS = ["n", "alpha", "sigma", "Lc", "t", "period", "depth", "duty"]

# (w_mtf, w_T, w_ca) settings spanning "sharpness-first" -> "brightness-first"
SWEEP = [(3.0, 0.3, 0.5), (2.0, 0.6, 0.5), (1.0, 1.0, 0.5),
         (0.6, 2.0, 0.5), (0.3, 3.0, 0.5), (1.0, 1.0, 2.0)]


def run_one(w_mtf, w_T, w_ca, seed=0):
    torch.manual_seed(seed)
    z0 = normalize_theta(sample_theta(N_STARTS)).clamp(1e-3, 1 - 1e-3)
    w = torch.log(z0 / (1 - z0)).requires_grad_(True)
    opt = torch.optim.Adam([w], lr=LR)
    for _ in range(N_STEPS):
        theta = denormalize_theta(torch.sigmoid(w))
        y = forward_model(theta)
        J = w_mtf * y[:, 0] + w_T * (y[:, 1] / 0.10) - w_ca * (y[:, 2] / 30.0)
        opt.zero_grad(); (-J.sum()).backward(); opt.step()
    with torch.no_grad():
        theta = denormalize_theta(torch.sigmoid(w))
        y = forward_model(theta)
        J = w_mtf * y[:, 0] + w_T * (y[:, 1] / 0.10) - w_ca * (y[:, 2] / 30.0)
        i = J.argmax()
    return theta[i], y[i]


def main():
    use_pmma()
    rows = []
    for w_mtf, w_T, w_ca in SWEEP:
        theta, y = run_one(w_mtf, w_T, w_ca)
        mtf, T, ca, T_fov = y.tolist()
        print(f"w=({w_mtf},{w_T},{w_ca}) -> MTF {mtf:.4f} | T {100*T:.2f}% | "
              f"chrom {ca:.2f}deg | period {theta[5]:.0f}nm depth {theta[6]:.0f}nm "
              f"duty {theta[7]:.2f} t {theta[4]:.2f}mm")
        rows.append([w_mtf, w_T, w_ca, mtf, T, ca, T_fov] + theta.tolist())

    with open("pareto_results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["w_mtf", "w_T", "w_ca", "MTF", "T", "chrom_deg", "T_fov"] + LABELS)
        wr.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mtfs = [r[3] for r in rows]; Ts = [100 * r[4] for r in rows]
        cas = [r[5] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sc = ax.scatter(Ts, mtfs, c=cas, cmap="viridis", s=90, edgecolors="k")
        for r, x, yv in zip(rows, Ts, mtfs):
            ax.annotate(f"({r[0]},{r[1]},{r[2]})", (x, yv), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        fig.colorbar(sc, label="chromatic spread (deg)")
        ax.set_xlabel("Transmission (%)"); ax.set_ylabel("System MTF @ 40 cyc/mm")
        ax.set_title("PMMA design trade-off frontier (placeholder physics)")
        fig.tight_layout(); fig.savefig("pareto_front.png", dpi=150)
        print("saved pareto_front.png")
    except ImportError:
        print("matplotlib not installed; CSV only")


if __name__ == "__main__":
    main()
