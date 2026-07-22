"""
Live self-updating dashboard + per-metric hall of fame for the metagrating
study. The optimizer calls update() every few iterations; this rewrites
metagrating_live.html (which auto-reloads itself every 3 s via meta-refresh),
and record() maintains metagrating_hof.json, the best-ever design per metric
(eta_TE, eta_TM, eta_unpol), across ALL runs and both arms. Open
metagrating_live.html in a browser and leave it open while optimizing.
"""

import json
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
# anchored to this folder (previously CWD-relative: running from the project
# root wrote a second, divergent set of dashboard/HOF files at the root)
HTML = os.path.join(_HERE, "metagrating_live.html")
HOF = os.path.join(_HERE, "metagrating_hof.json")
METRICS = ("eta_TE", "eta_TM", "eta_unpol")


def load_hof():
    if os.path.exists(HOF):
        with open(HOF) as f:
            return json.load(f)
    return {m: {"value": 0.0, "arm": "-", "when": "-", "detail": {}} for m in METRICS}


def record(metric, value, arm, detail):
    """Update the hall of fame if value beats the record. Returns True if new
    record. detail: JSON-serializable design description."""
    hof = load_hof()
    if value > hof.get(metric, {}).get("value", 0.0):
        hof[metric] = {"value": float(value), "arm": arm,
                       "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "detail": detail}
        with open(HOF, "w") as f:
            json.dump(hof, f, indent=1)
        return True
    return False


def _svg_history(hist, w=640, h=170):
    if len(hist) < 2:
        return "<i>waiting for iterations…</i>"
    top = max(max(hist), 1e-9)
    pts = " ".join(f"{20 + i*(w-30)/(len(hist)-1):.1f},"
                   f"{h-18 - v/top*(h-40):.1f}" for i, v in enumerate(hist))
    return (f'<svg width="{w}" height="{h}" style="background:#10161f;'
            f'border:1px solid #232c40;border-radius:6px">'
            f'<polyline points="{pts}" fill="none" stroke="#5aa9ff" stroke-width="1.6"/>'
            f'<text x="6" y="14" fill="#8b96ab" font-size="10">{100*top:.2f}%</text>'
            f'<text x="6" y="{h-6}" fill="#8b96ab" font-size="10">0</text></svg>')


def _svg_design(occ, w=640, h=130):
    """occupancy [NL,Nx] -> heatmap; 1 = PMMA, 0 = air."""
    if occ is None:
        return "<i>no design yet</i>"
    NL, NX = len(occ), len(occ[0])
    cw, ch = w / NX, (h - 20) / NL
    cells = []
    for k in range(NL):
        for i in range(NX):
            v = occ[k][i]
            if v > 0.02:
                cells.append(f'<rect x="{i*cw:.1f}" y="{k*ch:.1f}" width="{cw+.3:.2f}" '
                             f'height="{ch+.3:.2f}" fill="rgba(90,169,255,{min(v,1):.2f})"/>')
    return (f'<svg width="{w}" height="{h}" style="background:#10161f;'
            f'border:1px solid #232c40;border-radius:6px">'
            + "".join(cells) +
            f'<text x="4" y="{h-5}" fill="#8b96ab" font-size="10">one period → '
            f'(top = air side, bottom = PMMA substrate; blue = PMMA)</text></svg>')


def update(state):
    """state: dict(arm, period, seed, start, iteration, total_iters, beta,
    history[list of nominal eta], occ[[...]], energy_residual, note)."""
    hof = load_hof()
    rows = "".join(
        f"<tr><td>{m}</td><td><b>{100*hof[m]['value']:.3f}%</b></td>"
        f"<td>{hof[m]['arm']}</td><td>{hof[m]['when']}</td></tr>" for m in METRICS)
    hist = state.get("history", [])
    cur = hist[-1] if hist else 0.0
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="3">
<title>Metagrating optimization, LIVE</title>
<style>body{{background:#0d1117;color:#e8ecf5;font:13px -apple-system,sans-serif;
max-width:700px;margin:24px auto}} h1{{font-size:17px}} h2{{font-size:13px;color:#5aa9ff;
text-transform:uppercase;letter-spacing:.5px;margin-top:18px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
td,th{{border-bottom:1px solid #222c3d;padding:4px 8px;text-align:left}}
.big{{font-size:26px;font-weight:700;color:#59d98c}}
.dim{{color:#8b96ab}}</style></head><body>
<h1>PMMA metagrating optimization, live <span class="dim">(auto-refreshes)</span></h1>
<div class="dim">{time.strftime("%H:%M:%S")} · arm: <b>{state.get('arm','-')}</b> ·
Λ = {state.get('period','-')} nm · seed {state.get('seed','-')} ·
start {state.get('start','-')} · β = {state.get('beta','-')} ·
iter {state.get('iteration',0)}/{state.get('total_iters','?')} ·
|R+T−1| = {state.get('energy_residual',0):.1e}</div>
<h2>Current nominal η₋₁ (TE)</h2>
<div class="big">{100*cur:.3f}%</div>
{_svg_history(hist)}
<h2>Current design (one period, ρ(x,z))</h2>
{_svg_design(state.get('occ'))}
<h2>Hall of fame, best ever, all runs</h2>
<table><tr><th>metric</th><th>record</th><th>arm</th><th>when</th></tr>{rows}</table>
<div class="dim" style="margin-top:8px">{state.get('note','')}</div>
</body></html>"""
    with open(HTML, "w") as f:
        f.write(html)
