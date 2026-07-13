"""One map of the project folders, so every script reads and writes the same
organized tree no matter where it is launched from.

    docs/         how-to guides, objectives, citations, reviews
    physics/      the differentiable physics engine + rigorous RCWA + tests
    networks/     the neural networks (surrogate, tandem inverse, adjoint search)
    baselines/    non-neural gradient searches the paper compares against
    visuals/      the scripts that build the 3D model and the HTML report
    figures/      auto-updating charts (PNG)
    results/      tables, winners, records (CSV) + the printable 3D mesh
    checkpoints/  trained network weights (.pt)

The two double-click pages (results_report.html, waveguide_3d.html) live in
the project root next to overnight.sh.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")
CHECKPOINTS_DIR = os.path.join(ROOT, "checkpoints")
for _d in (FIGURES_DIR, RESULTS_DIR, CHECKPOINTS_DIR):
    os.makedirs(_d, exist_ok=True)


def fig_path(name: str) -> str:
    return os.path.join(FIGURES_DIR, name)


def res_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, name)


def ckpt_path(name: str) -> str:
    return os.path.join(CHECKPOINTS_DIR, name)


def root_path(name: str) -> str:
    return os.path.join(ROOT, name)
