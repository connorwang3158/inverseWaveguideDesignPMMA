"""
Build a viewable 3D model of the current best PMMA waveguide design.

Reads the winning design (optimal_designs_na.csv from the neural-adjoint
search, falling back to best_design_ever_v2.csv, falling back to sensible
defaults), then writes:

  waveguide_3d.html    interactive 3D viewer — rotate/zoom with the mouse,
                       drag sliders to change the design, watch the RGB ray
                       paths and grating geometry update live. Fully
                       self-contained (its own software renderer, no internet
                       needed) — just double-click it.
  waveguide_model.stl  the same geometry as a standard STL mesh — opens in
                       any 3D viewer/slicer (macOS Quick Look, Windows 3D
                       Viewer, Blender, PrusaSlicer) and is 3D-printable

Scale note: real grating teeth are a few hundred NANOMETERS — invisible next
to a millimeter slab — so the couplers are drawn with the tooth geometry
magnified by GRATING_ZOOM (labeled in the viewer). Proportions within the
grating (period : depth : duty) are true to the design.

Usage:  python3 make_3d_model.py
"""

import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import res_path, root_path

# slab footprint (mm) — typical AR-combiner eyepiece scale
SLAB_L, SLAB_W = 30.0, 12.0
COUPLER_L, COUPLER_W = 6.0, 8.0     # grating patch footprint (mm)
GRATING_ZOOM = 2000.0               # nm-scale teeth magnified for visibility

DEFAULTS = {"n": 1.49, "t_mm": 1.0, "period_nm": 510.0, "depth_nm": 330.0,
            "duty": 0.5}


def load_best_design():
    """Pull the winning design out of the result CSVs, if any exist."""
    cols = {"t_mm": "t(mm)", "period_nm": "period(nm)",
            "depth_nm": "depth(nm)", "duty": "duty", "n": "n"}
    for name in ("optimal_designs_na.csv", "best_design_ever_v3.csv",
                 "best_design_ever_v2.csv"):
        path = res_path(name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            d = {k: float(rows[0][c]) for k, c in cols.items()}
            print(f"[3d] using best design from results/{name}: "
                  f"period {d['period_nm']:.0f} nm, depth {d['depth_nm']:.0f} nm, "
                  f"duty {d['duty']:.2f}, t {d['t_mm']:.2f} mm")
            return d
    print("[3d] no result CSVs found — using default design "
          "(run neural_adjoint.py to feed in a real winner)")
    return dict(DEFAULTS)


# ---------------------------------------------------------------------------
# STL export (ASCII) — boxes only: slab + magnified grating teeth
# ---------------------------------------------------------------------------

def _box_facets(cx, cy, cz, sx, sy, sz):
    """12 triangles for an axis-aligned box centered at (cx,cy,cz)."""
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    v = {0: (x0, y0, z0), 1: (x1, y0, z0), 2: (x1, y1, z0), 3: (x0, y1, z0),
         4: (x0, y0, z1), 5: (x1, y0, z1), 6: (x1, y1, z1), 7: (x0, y1, z1)}
    quads = [((0, 3, 2, 1), (0, 0, -1)), ((4, 5, 6, 7), (0, 0, 1)),
             ((0, 1, 5, 4), (0, -1, 0)), ((2, 3, 7, 6), (0, 1, 0)),
             ((1, 2, 6, 5), (1, 0, 0)), ((0, 4, 7, 3), (-1, 0, 0))]
    facets = []
    for (a, b, c, d), nrm in quads:
        facets.append((nrm, v[a], v[b], v[c]))
        facets.append((nrm, v[a], v[c], v[d]))
    return facets


def write_stl(design, path=None):
    if path is None:
        path = res_path("waveguide_model.stl")
    t = design["t_mm"]
    period_mm = design["period_nm"] * GRATING_ZOOM * 1e-6
    depth_mm = design["depth_nm"] * GRATING_ZOOM * 1e-6
    duty = design["duty"]

    facets = _box_facets(0, 0, t / 2, SLAB_L, SLAB_W, t)   # slab, top at z=t
    for x_center in (-SLAB_L / 2 + 1 + COUPLER_L / 2,      # in-coupler
                     SLAB_L / 2 - 1 - COUPLER_L / 2):      # out-coupler
        n_teeth = max(int(COUPLER_L / period_mm), 1)
        for i in range(n_teeth):
            x = x_center - COUPLER_L / 2 + (i + 0.5) * period_mm
            facets += _box_facets(x, 0, t + depth_mm / 2,
                                  period_mm * duty, COUPLER_W, depth_mm)

    with open(path, "w") as f:
        f.write("solid pmma_waveguide\n")
        for (nx, ny, nz), p1, p2, p3 in facets:
            f.write(f" facet normal {nx} {ny} {nz}\n  outer loop\n")
            for px, py, pz in (p1, p2, p3):
                f.write(f"   vertex {px:.5f} {py:.5f} {pz:.5f}\n")
            f.write("  endloop\n endfacet\n")
        f.write("endsolid pmma_waveguide\n")
    print(f"[3d] wrote {path} ({len(facets)} triangles, units mm, "
          f"grating x{GRATING_ZOOM:.0f})")


# ---------------------------------------------------------------------------
# Interactive HTML viewer — self-contained painter's-algorithm 3D renderer,
# so the page works with no internet and no libraries, on any computer.
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PMMA Waveguide — 3D Model</title>
<style>
  :root { --bg:#10141c; --panel:#1a2130; --ink:#e8ecf5; --dim:#8b96ab; --acc:#5aa9ff; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--ink); font:14px/1.5 -apple-system,Helvetica,sans-serif; overflow:hidden; }
  canvas { display:block; cursor:grab; }
  canvas:active { cursor:grabbing; }
  .panel { position:fixed; top:14px; left:14px; width:295px; background:var(--panel);
    border-radius:10px; padding:14px; opacity:.96; max-height:calc(100vh - 28px); overflow-y:auto; }
  h1 { font-size:17px; } .sub { color:var(--dim); font-size:11.5px; margin:2px 0 10px; }
  label { display:block; font-size:12px; color:var(--dim); margin-top:9px; }
  label b { color:var(--ink); float:right; }
  input[type=range] { width:100%; accent-color:var(--acc); }
  .metrics { margin-top:12px; border-top:1px solid #33405a; padding-top:8px; }
  .m { display:flex; justify-content:space-between; font-size:12px; margin-top:4px; }
  .m span:first-child { color:var(--dim); }
  .ok { color:#37d095; } .bad { color:#ff7a7a; }
  .note { font-size:11px; color:var(--dim); margin-top:10px; }
</style>
</head>
<body>
<canvas id="cv"></canvas>
<div class="panel">
  <h1>PMMA Waveguide — 3D Model</h1>
  <div class="sub">Drag to rotate &middot; scroll to zoom.
  Grating teeth magnified &times;__ZOOM__ (period:depth:duty proportions are true).</div>
  <label>Grating period &Lambda; <b id="v_per"></b></label>
  <input type="range" id="per" min="300" max="700" step="1">
  <label>Grating depth d <b id="v_dep"></b></label>
  <input type="range" id="dep" min="20" max="400" step="1">
  <label>Duty cycle <b id="v_dut"></b></label>
  <input type="range" id="dut" min="20" max="80" step="1">
  <label>Slab thickness t <b id="v_thk"></b></label>
  <input type="range" id="thk" min="30" max="200" step="1">
  <div class="metrics">
    <div class="m"><span>1st-order angle (green, in-guide)</span><b id="m_ang"></b></div>
    <div class="m"><span>Guided by TIR? (needs &gt; 42.2&deg;)</span><b id="m_tir"></b></div>
    <div class="m"><span>Chromatic spread (red&minus;blue)</span><b id="m_ca"></b></div>
    <div class="m"><span>Scalar coupling efficiency &eta;&#8321;</span><b id="m_eta"></b></div>
    <div class="m"><span>TIR bounces across the slab</span><b id="m_bn"></b></div>
  </div>
  <div class="note">Rays: blue 450 nm, green 532 nm, red 635 nm — angles from the
  in-guide grating equation n&thinsp;sin&theta; = &lambda;/&Lambda; (n = 1.49). The RGB fan
  widening with propagation IS the chromatic-spread metric the networks optimize.
  Sliders start at the current best design found by the neural-adjoint search.</div>
</div>
<script>
"use strict";
const DESIGN = __DESIGN_JSON__;
const ZOOM = __ZOOM__;
const SLAB_L = __SLAB_L__, SLAB_W = __SLAB_W__;
const COUP_L = __COUP_L__, COUP_W = __COUP_W__;
const N_INDEX = 1.49, WL = [450, 532, 635];
const RAYCOL = ["#5a8cff", "#37d095", "#ff6b5a"];

const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
let yaw = 0.65, pitch = 0.42, dist = 46;
let faces = [], lines = [];        // world-space geometry, rebuilt on slider move

// ---- tiny 3D engine: rotate -> perspective project -> painter-sort quads ----
function project(p) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  let x = p[0] * cy + p[2] * sy, z = -p[0] * sy + p[2] * cy, y = p[1];
  let y2 = y * cp - z * sp, z2 = y * sp + z * cp;
  z2 += dist;
  const f = 900 / Math.max(z2, 1);
  return [cv.width / 2 + x * f, cv.height / 2 - y2 * f, z2];
}
function shade(hex, k) {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  return `rgb(${Math.round(r * k)},${Math.round(g * k)},${Math.round(b * k)})`;
}
function normal(q) {
  const u = [q[1][0]-q[0][0], q[1][1]-q[0][1], q[1][2]-q[0][2]];
  const v = [q[3][0]-q[0][0], q[3][1]-q[0][1], q[3][2]-q[0][2]];
  const n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]];
  const L = Math.hypot(...n) || 1;
  return n.map(c => c / L);
}
const LIGHT = (() => { const l = [0.5, 0.8, 0.35], L = Math.hypot(...l);
                       return l.map(c => c / L); })();

function box(cx, cy, cz, sx, sy, sz, color, alpha) {
  const x0=cx-sx/2, x1=cx+sx/2, y0=cy-sy/2, y1=cy+sy/2, z0=cz-sz/2, z1=cz+sz/2;
  const V = [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
             [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]];
  for (const idx of [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]])
    faces.push({ q: idx.map(i => V[i]), color, alpha });
}

function draw() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  // floor grid
  ctx.strokeStyle = "#232c40"; ctx.lineWidth = 1;
  for (let i = -30; i <= 30; i += 5) {
    for (const seg of [[[i,-4,-30],[i,-4,30]], [[-30,-4,i],[30,-4,i]]]) {
      const a = project(seg[0]), b = project(seg[1]);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
  }
  // faces back-to-front (painter's algorithm)
  const drawable = faces.map(f => {
    const pts = f.q.map(project);
    const zAvg = (pts[0][2] + pts[1][2] + pts[2][2] + pts[3][2]) / 4;
    const n = normal(f.q);
    const lambert = Math.abs(n[0]*LIGHT[0] + n[1]*LIGHT[1] + n[2]*LIGHT[2]);
    return { pts, zAvg, fill: shade(f.color, 0.35 + 0.65 * lambert), alpha: f.alpha };
  }).sort((a, b) => b.zAvg - a.zAvg);
  for (const f of drawable) {
    ctx.globalAlpha = f.alpha;
    ctx.fillStyle = f.fill;
    ctx.beginPath();
    ctx.moveTo(f.pts[0][0], f.pts[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(f.pts[i][0], f.pts[i][1]);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = "rgba(10,14,22,.35)"; ctx.lineWidth = 0.6; ctx.stroke();
  }
  ctx.globalAlpha = 1;
  // ray polylines on top
  for (const ln of lines) {
    ctx.strokeStyle = ln.color; ctx.lineWidth = 1.6; ctx.globalAlpha = 0.95;
    ctx.beginPath();
    ln.pts.forEach((p, i) => {
      const s = project(p);
      i ? ctx.lineTo(s[0], s[1]) : ctx.moveTo(s[0], s[1]);
    });
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

// ---- scene construction from the design parameters ----
function rebuild(p) {
  faces = []; lines = [];
  const t = p.t_mm;
  box(0, t / 2, 0, SLAB_L, t, SLAB_W, "#7db8e8", 0.32);       // PMMA slab

  const periodMm = p.period_nm * ZOOM * 1e-6;
  const depthMm = p.depth_nm * ZOOM * 1e-6;
  const nTeeth = Math.max(1, Math.floor(COUP_L / periodMm));
  const centers = [-SLAB_L / 2 + 1 + COUP_L / 2, SLAB_L / 2 - 1 - COUP_L / 2];
  for (const cx of centers)                                    // grating teeth
    for (let i = 0; i < nTeeth; i++)
      box(cx - COUP_L / 2 + (i + 0.5) * periodMm, t + depthMm / 2, 0,
          periodMm * p.duty, depthMm, COUP_W, "#5aa9ff", 1.0);

  const [xIn, xOut] = centers;                                 // RGB ray paths
  for (let k = 0; k < 3; k++) {
    const s = WL[k] / (N_INDEX * p.period_nm);
    if (s >= 1) continue;                                      // evanescent
    const th = Math.asin(s), step = t * Math.tan(th);
    const zOff = (k - 1) * 0.3;                                // separate the 3 rays
    const pts = [[xIn, t + 6, zOff], [xIn, t, zOff]];
    let x = xIn, down = true;
    while (x + step < xOut && pts.length < 400) {
      x += step;
      pts.push([x, down ? 0 : t, zOff]);
      down = !down;
    }
    pts.push([xOut, down ? 0 : t, zOff]);
    if (!down) pts.push([xOut, 0, zOff]);
    pts.push([xOut + 1.5, -5, zOff]);                          // toward the eye
    lines.push({ pts, color: RAYCOL[k] });
  }

  // metrics readout (same formulas as waveguide_physics.py)
  const thG = Math.asin(Math.min(532 / (N_INDEX * p.period_nm), 0.999));
  const angB = Math.asin(Math.min(450 / (N_INDEX * p.period_nm), 0.999));
  const angR = Math.asin(Math.min(635 / (N_INDEX * p.period_nm), 0.999));
  const tir = thG > Math.asin(1 / N_INDEX);
  const phi = 2 * Math.PI * p.depth_nm * (N_INDEX - 1) / 532;
  const eta = 4 * (Math.sin(Math.PI * p.duty) / Math.PI) ** 2
                * Math.sin(phi / 2) ** 2;
  const bounces = Math.max(0, Math.floor((xOut - xIn) / (t * Math.tan(thG))));
  const deg = r => (r * 180 / Math.PI).toFixed(1) + "°";
  document.getElementById("m_ang").textContent = deg(thG);
  const tEl = document.getElementById("m_tir");
  tEl.textContent = tir ? "yes" : "NO — light escapes";
  tEl.className = tir ? "ok" : "bad";
  document.getElementById("m_ca").textContent = deg(angR - angB);
  document.getElementById("m_eta").textContent = (100 * eta).toFixed(1) + "%";
  document.getElementById("m_bn").textContent = bounces;
  draw();
}

// ---- UI wiring ----
const sliders = { per: "period_nm", dep: "depth_nm", dut: "duty", thk: "t_mm" };
const toUi = { per: v => v, dep: v => v, dut: v => v * 100, thk: v => v * 100 };
const fromUi = { per: v => v, dep: v => v, dut: v => v / 100, thk: v => v / 100 };
const fmt = { per: v => v.toFixed(0) + " nm", dep: v => v.toFixed(0) + " nm",
              dut: v => v.toFixed(2), thk: v => v.toFixed(2) + " mm" };
const state = Object.assign({}, DESIGN);
function sync() {
  for (const id in sliders)
    document.getElementById("v_" + id).textContent = fmt[id](state[sliders[id]]);
  rebuild(state);
}
for (const id in sliders) {
  const inp = document.getElementById(id);
  inp.value = toUi[id](DESIGN[sliders[id]]);
  inp.addEventListener("input", () => {
    state[sliders[id]] = fromUi[id](parseFloat(inp.value)); sync();
  });
}

let dragging = false, px = 0, py = 0;
cv.addEventListener("mousedown", e => { dragging = true; px = e.clientX; py = e.clientY; });
addEventListener("mouseup", () => dragging = false);
addEventListener("mousemove", e => {
  if (!dragging) return;
  yaw += (e.clientX - px) * 0.008; pitch += (e.clientY - py) * 0.008;
  pitch = Math.max(-1.4, Math.min(1.4, pitch));
  px = e.clientX; py = e.clientY; draw();
});
cv.addEventListener("wheel", e => {
  e.preventDefault();
  dist = Math.max(12, Math.min(150, dist * (1 + Math.sign(e.deltaY) * 0.08)));
  draw();
}, { passive: false });
cv.addEventListener("touchstart", e => { px = e.touches[0].clientX; py = e.touches[0].clientY; });
cv.addEventListener("touchmove", e => {
  e.preventDefault();
  yaw += (e.touches[0].clientX - px) * 0.008;
  pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.touches[0].clientY - py) * 0.008));
  px = e.touches[0].clientX; py = e.touches[0].clientY; draw();
}, { passive: false });

function resize() {
  cv.width = innerWidth * devicePixelRatio; cv.height = innerHeight * devicePixelRatio;
  cv.style.width = innerWidth + "px"; cv.style.height = innerHeight + "px";
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  cv.width = innerWidth; cv.height = innerHeight;   // draw in CSS pixels
  draw();
}
addEventListener("resize", resize);
resize();
sync();
</script>
</body>
</html>
"""


def write_html(design, path=None):
    if path is None:
        path = root_path("waveguide_3d.html")   # double-click page stays at root
    html = (HTML_TEMPLATE
            .replace("__DESIGN_JSON__", json.dumps(design))
            .replace("__ZOOM__", f"{GRATING_ZOOM:.0f}")
            .replace("__SLAB_L__", str(SLAB_L)).replace("__SLAB_W__", str(SLAB_W))
            .replace("__COUP_L__", str(COUPLER_L)).replace("__COUP_W__", str(COUPLER_W)))
    with open(path, "w") as f:
        f.write(html)
    print(f"[3d] wrote {path} (double-click to open in a browser; works offline)")


if __name__ == "__main__":
    d = load_best_design()
    write_html(d)
    write_stl(d)
    # sanity: the viewer's TIR condition should hold for the loaded design
    s = 532.0 / (d["n"] * d["period_nm"])
    if s < 1 and math.asin(s) > math.asin(1 / d["n"]):
        print("[3d] check: green ray is TIR-guided for this design ✔")
    else:
        print("[3d] WARNING: green first-order is NOT TIR-guided for this design")
