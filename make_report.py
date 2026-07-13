"""
Compile all result CSVs and figures into one readable HTML report.

Usage:  python3 make_report.py     ->  results_report.html  (double-click to open)
"""

import csv
import os
from datetime import datetime

FILES = [
    ("Forward surrogate training runs", "surrogate_runs.csv"),
    ("Inverse network training runs (5-seed protocol table)", "training_runs.csv"),
    ("Neural-adjoint winners (network-found, physics-verified)", "optimal_designs_na.csv"),
    ("Direct gradient-search winners (physics baseline)", "optimal_designs.csv"),
    ("All-time best design (hall of fame)", "best_design_ever.csv"),
    ("Trade-off sweep (priority menu)", "pareto_results.csv"),
    ("Scalar vs rigorous RCWA validation", "rcwa_validation.csv"),
]
IMAGES = [("Forward surrogate training curve", "surrogate_loss_curve.png"),
          ("Surrogate parity (network vs exact physics)", "surrogate_parity.png"),
          ("Inverse network training curve", "loss_curve.png"),
          ("Neural-adjoint search", "neural_adjoint_run.png"),
          ("Trade-off frontier", "pareto_front.png")]


def table_html(path):
    with open(path) as f:
        rows = list(csv.reader(f))
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
<p>Generated {datetime.now():%Y-%m-%d %H:%M}. Re-run <code>python3 make_report.py</code>
after any experiment to refresh.</p>"""]
    for title, path in FILES:
        parts.append(f"<h2>{title}</h2>")
        parts.append(table_html(path) if os.path.exists(path)
                     else f"<p class='miss'>{path} not found — run the matching script first.</p>")
    for title, path in IMAGES:
        parts.append(f"<h2>{title}</h2>")
        parts.append(f"<img src='{path}'>" if os.path.exists(path)
                     else f"<p class='miss'>{path} not found.</p>")
    parts.append("<h2>3D model</h2>")
    parts.append("<p><a href='waveguide_3d.html'>waveguide_3d.html</a> — interactive "
                 "3D waveguide (rotate/zoom, live sliders); <code>waveguide_model.stl"
                 "</code> opens in any 3D viewer. Regenerate with "
                 "<code>python3 make_3d_model.py</code>."
                 if os.path.exists("waveguide_3d.html")
                 else "<p class='miss'>waveguide_3d.html not found — run "
                      "python3 make_3d_model.py.</p>")
    parts.append("</body></html>")
    with open("results_report.html", "w") as f:
        f.write("\n".join(parts))
    print("written -> results_report.html (double-click it)")


if __name__ == "__main__":
    main()
