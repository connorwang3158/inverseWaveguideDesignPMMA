"""
Live-visualized design optimization: watch the model work in real time.

Opens a matplotlib window that updates every 10 gradient steps while the
optimizer climbs: left panel = metric trajectories; right panel = the current
best waveguide drawn to scale with its grating.

Usage:  python3 optimize_live.py          (requires: pip3 install matplotlib)
"""

import os
import sys

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.waveguide_physics import (
    forward_model, use_pmma, sample_theta, normalize_theta, denormalize_theta,
)

N_STARTS, N_STEPS, LR = 120, 400, 0.03
W_MTF, W_T, W_CA = 1.0, 1.0, 0.5


def objective(y):
    return W_MTF * y[:, 0] + W_T * (y[:, 1] / 0.10) - W_CA * (y[:, 2] / 30.0)


def draw_design(ax, theta, y):
    ax.clear()
    per, dep, dut, t = theta[5].item(), theta[6].item(), theta[7].item(), theta[4].item()
    ax.add_patch(plt.Rectangle((0, 0), 10, t, color="#9fc5ff", alpha=0.6))
    n_teeth, w = 30, 10 / 30
    for i in range(n_teeth):                       # grating teeth (scaled)
        ax.add_patch(plt.Rectangle((i * w, t), w * dut, dep * 1e-3, color="#1f5fbf"))
    ax.set_xlim(0, 10); ax.set_ylim(0, max(t * 1.6, t + 0.5))
    ax.set_title(f"best design now:  Λ={per:.0f}nm  d={dep:.0f}nm  duty={dut:.2f}  "
                 f"t={t:.2f}mm\nMTF {y[0]:.3f} | T {100*y[1]:.2f}% | "
                 f"chrom {y[2]:.2f}° | T@FOV {100*y[3]:.2f}%")
    ax.set_xlabel("waveguide length (arb.)"); ax.set_ylabel("mm")


def main():
    use_pmma(); torch.manual_seed(0)
    z0 = normalize_theta(sample_theta(N_STARTS)).clamp(1e-3, 1 - 1e-3)
    w = torch.log(z0 / (1 - z0)).requires_grad_(True)
    opt = torch.optim.Adam([w], lr=LR)

    plt.ion()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    hist = {"step": [], "MTF": [], "T": [], "CA": []}

    for step in range(N_STEPS):
        theta = denormalize_theta(torch.sigmoid(w))
        y = forward_model(theta)
        J = objective(y)
        opt.zero_grad(); (-J.sum()).backward(); opt.step()

        if step % 10 == 0 or step == N_STEPS - 1:
            with torch.no_grad():
                i = J.argmax()
                hist["step"].append(step)
                hist["MTF"].append(y[i, 0].item())
                hist["T"].append(y[i, 1].item() * 100)
                hist["CA"].append(y[i, 2].item())
                axL.clear()
                axL.plot(hist["step"], hist["MTF"], label="MTF")
                axL.plot(hist["step"], [v / 15 for v in hist["T"]], label="T (%/15)")
                axL.plot(hist["step"], [v / 15 for v in hist["CA"]], label="chrom (°/15)")
                axL.set_xlabel("gradient step"); axL.legend(loc="lower right")
                axL.set_title(f"optimization progress (step {step}/{N_STEPS})")
                draw_design(axR, theta[i], y[i])
                fig.canvas.draw(); fig.canvas.flush_events(); plt.pause(0.01)

    plt.ioff()
    print("done — close the window to exit; final design shown on the right")
    plt.show()


if __name__ == "__main__":
    main()
