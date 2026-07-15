"""
Compile all result CSVs and figures into one readable HTML report.

Usage:  python3 visuals/make_report.py   ->  results_report.html at the
project root (double-click to open). Tables come from results/, images
from figures/.
"""

import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import fig_path, res_path, root_path

FILES = [
    ("Forward surrogate runs (physics-learning table)", "surrogate_runs.csv"),
    ("Inverse-network training runs (5-seed protocol table)", "training_runs.csv"),
    ("Neural-adjoint winners (network-found, physics-verified)", "optimal_designs_na.csv"),
    ("All-time best design (corrected physics)", "best_design_ever_v2.csv"),
    ("Optimal designs (gradient-baseline winners)", "optimal_designs.csv"),
    ("Trade-off sweep (priority menu)", "pareto_results.csv"),
    ("Scalar vs rigorous RCWA validation", "rcwa_validation.csv"),
    ("RCWA check of neural-adjoint winners (scalar-validity audit)",
     "design_rcwa_check_na.csv"),
    ("Memorization audit (numbers)", "memorization_audit.csv"),
]
IMAGES = [("Surrogate learning curve", "surrogate_loss_curve.png"),
          ("Surrogate parity (prediction vs exact physics)", "surrogate_parity.png"),
          ("Memorization audit (generalization evidence)", "memorization_audit.png"),
          ("Inverse-network training curve", "loss_curve.png"),
          ("Neural-adjoint search", "neural_adjoint_run.png"),
          ("Trade-off frontier", "pareto_front.png")]


def table_html(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:   # empty file (e.g. a run interrupted mid-write) — don't crash
        return "<p class='miss'>file is empty.</p>"
    head = "".join(f"<th>{c}</th>" for c in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                   for r in rows[1:])
    return f"<table><tr>{head}</tr>{body}</table>"


def main():
    parts = [f"""<html><head><meta charset='utf-8'><title>Waveguide Results</title>
<style>body{{font:14px -apple-system,sans-serif;max-width:900px;margin:30px auto;color:#1a2233}}
h1{{font-size:22px}} h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #ccd;padding-bottom:4px}}
table{{border-collapse:collapse;margin-top:8px}} td,th{{border:1px solid #ccd;padding:4px 9px;font-size:12px}}
th{{background:#eef2fa}} img{{max-width:100%;border:1px solid #ccd;border-radius:6px;margin-top:8px}}
.miss{{color:#a33;font-size:12px}}</style></head><body>
<h1>AR Waveguide Inverse Design — Results Report</h1>
<p>Generated {datetime.now():%Y-%m-%d %H:%M}. Re-run <code>python3 visuals/make_report.py</code>
after any experiment to refresh.</p>"""]
    for title, name in FILES:
        parts.append(f"<h2>{title}</h2>")
        parts.append(table_html(res_path(name)) if os.path.exists(res_path(name))
                     else f"<p class='miss'>results/{name} not found — run the matching script first.</p>")
    for title, name in IMAGES:
        parts.append(f"<h2>{title}</h2>")
        # the report sits at the project root, so images resolve via figures/
        parts.append(f"<img src='figures/{name}'>" if os.path.exists(fig_path(name))
                     else f"<p class='miss'>figures/{name} not found.</p>")
    parts.append("<h2>3D model</h2>")
    if os.path.exists(root_path("waveguide_3d.html")):
        parts.append("<p>Open <a href='waveguide_3d.html'>waveguide_3d.html</a> "
                     "to rotate the winning waveguide in 3D (drag to rotate, "
                     "scroll to zoom). A printable mesh is in "
                     "<code>results/waveguide_model.stl</code>.</p>")
    else:
        parts.append("<p class='miss'>waveguide_3d.html not found — run "
                     "python3 visuals/make_3d_model.py.</p>")
    parts.append("</body></html>")
    with open(root_path("results_report.html"), "w") as f:
        f.write("\n".join(parts))
    print("written -> results_report.html at the project root (double-click it)")


if __name__ == "__main__":
    main()
