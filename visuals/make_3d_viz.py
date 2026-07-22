"""
Generate waveguide_designs_3d.html, an interactive Three.js visualization of
the top optimized PMMA waveguide designs.

Reads:  optimal_designs.csv      (from optimize_pmma.py, physics v2 columns)
        design_rcwa_check.csv    (optional, from rigorous_solver.py --designs)
Writes: waveguide_designs_3d.html (self-contained; double-click to open)

The page embeds a JavaScript port of the v2-ERA scalar engine (field-angle
grating equation, TIR guiding window, polarization-resolved Fresnel, v2
bounce count, Tien roughness, scalar grating efficiency) for interactive
ILLUSTRATION ONLY, it does NOT match the current v5 Python engine (RCWA
coupling, Watson eye MTF, walk-off chromatics, re-interaction term are not
ported). Quotable numbers come from the CSVs and the Python engine; RCWA
TE/TM efficiencies (exact vector Maxwell solutions) are shown alongside
where available.

Usage:  python3 make_3d_viz.py
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import res_path

DESIGNS_CSV = res_path("optimal_designs.csv")
RCWA_CSV = res_path("design_rcwa_check.csv")
OUT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "waveguide_designs_3d.html")


def load_designs():
    if not os.path.exists(DESIGNS_CSV):
        raise SystemExit(f"{DESIGNS_CSV} not found, run optimize_pmma.py first")
    with open(DESIGNS_CSV) as f:
        rows = list(csv.DictReader(f))
    designs = []
    for r in rows:
        designs.append({
            "rank": int(r["rank"]), "J": float(r["J"]),
            "MTF": float(r["MTF"]), "T": float(r["T"]),
            # v5 CSVs carry walkoff_mm; older archives carry chrom_deg
            "chrom_deg": float(r.get("walkoff_mm", r.get("chrom_deg", 0))),
            "T_fov": float(r["T_fov"]),
            "T_TE": float(r.get("T_TE", 0)), "T_TM": float(r.get("T_TM", 0)),
            "fov_lo": float(r.get("fov_lo_deg", 0)),
            "fov_hi": float(r.get("fov_hi_deg", 0)),
            "n": float(r["n"]), "alpha": float(r["alpha(1/mm)"]),
            "sigma": float(r["sigma(nm)"]), "Lc": float(r["Lc(nm)"]),
            "t": float(r["t(mm)"]), "period": float(r["period(nm)"]),
            "depth": float(r["depth(nm)"]), "duty": float(r["duty"]),
        })
    return designs


def load_rcwa():
    if not os.path.exists(RCWA_CSV):
        return {}
    out = {}
    with open(RCWA_CSV) as f:
        for r in csv.DictReader(f):
            out.setdefault(r["rank"], []).append({
                "lam": float(r["wavelength_nm"]),
                "scalar": float(r["scalar_eta1"]),
                "TE": float(r["rcwa_TE"]), "TM": float(r["rcwa_TM"]),
            })
    return out


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PMMA AR Waveguide, Top Designs in 3D</title>
<style>
  :root { --bg:#0d1117; --panel:#161d29; --ink:#e8ecf5; --dim:#8b96ab; --acc:#5aa9ff; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--ink); font:13px/1.45 -apple-system,Helvetica,sans-serif; overflow:hidden; }
  #scene { position:fixed; inset:0; }
  .hud { position:fixed; top:12px; left:12px; width:330px; background:rgba(22,29,41,.94);
         border:1px solid #2a3648; border-radius:10px; padding:12px 14px; max-height:96vh; overflow:auto; }
  h1 { font-size:15px; margin-bottom:2px; } .sub { color:var(--dim); font-size:11px; margin-bottom:8px; }
  label { display:block; font-size:11px; color:var(--dim); margin-top:8px; }
  select,input[type=range] { width:100%; accent-color:var(--acc); }
  select { padding:5px; border-radius:6px; border:1px solid #33405a; background:#232c40; color:var(--ink); }
  .seg { display:flex; gap:4px; margin-top:4px; }
  .seg button { flex:1; padding:5px 0; border-radius:6px; border:1px solid #33405a; background:#232c40;
                color:var(--dim); cursor:pointer; font-size:12px; }
  .seg button.on { background:#2b4a75; color:#fff; border-color:var(--acc); }
  table { width:100%; border-collapse:collapse; margin-top:8px; font-size:11.5px; }
  td { padding:2.5px 4px; border-bottom:1px solid #222c3d; } td:last-child { text-align:right; font-weight:600; }
  .good { color:#59d98c; } .warn { color:#ffb84d; } .bad { color:#ff6b5a; }
  .eq { background:#10161f; border:1px solid #232c40; border-radius:6px; padding:7px 9px; margin-top:8px;
        font:11px/1.7 "SF Mono",Menlo,monospace; color:#b9c4d8; white-space:pre; overflow-x:auto; }
  .cap { font-size:10.5px; color:var(--dim); margin-top:6px; }
  #angplot { background:#10161f; border:1px solid #232c40; border-radius:6px; margin-top:8px; width:100%; }
  h2 { font-size:12px; margin-top:12px; color:var(--acc); text-transform:uppercase; letter-spacing:.5px; }
  .tag { position:fixed; right:14px; bottom:10px; color:var(--dim); font-size:11px; text-align:right; }
</style>
</head>
<body>
<div id="scene"></div>
<div class="hud">
  <h1>PMMA AR Waveguide, Top Designs (3D)</h1>
  <div class="sub">Illustrative v2-era scalar physics (live JS); quotable numbers
  come from the v5 Python engine + RCWA CSVs. Drag = orbit, wheel = zoom.
  Grating relief exaggerated ×2000 for visibility.</div>

  <label>Design (gradient-search winners)</label>
  <select id="design"></select>

  <label>Polarization</label>
  <div class="seg" id="polseg">
    <button data-p="unpol" class="on">unpolarized</button>
    <button data-p="TE">TE (s)</button>
    <button data-p="TM">TM (p)</button>
  </div>

  <label>Field angle θᵢ <b id="v_fov" style="float:right;color:var(--ink)"></b></label>
  <input type="range" id="fov" min="-20" max="20" value="0" step="0.5">

  <h2>Performance (analytic engine, live)</h2>
  <table id="perf"></table>

  <h2>RCWA vector verification (Maxwell, exact)</h2>
  <table id="rcwa"></table>
  <div class="cap" id="rcwacap"></div>

  <h2>Governing equations</h2>
  <div class="eq">in-coupling (m=+1):  n·sinθ_d = sinθᵢ + λ/Λ
TIR guiding window:  1 &lt; sinθᵢ + λ/Λ &lt; n
Fresnel TE: r = (cosθᵢ − n·cosθ_t)/(cosθᵢ + n·cosθ_t)
Fresnel TM: r = (n·cosθᵢ − cosθ_t)/(n·cosθᵢ + cosθ_t)
bounces:  N_b = L/(2·t·tanθ_d)
Tien/bounce: exp[−(4πσ·n·cosθ_d/λ)²·S(L_c)]
grating:  η₁ = 4(sin(πD)/π)²·sin²(φ/2), φ = 2πd(n−1)/λ
T = TIR·T_F²·e^(−αℓ)·T_scat·η_in(θ)·η_out</div>

  <h2>Transmission vs field angle</h2>
  <canvas id="angplot" width="300" height="150"></canvas>
  <div class="cap">Solid: TE · dashed: TM · thick: unpolarized. Vertical band =
  full-RGB guided FOV window (the index-limited FOV of the waveguide).</div>
</div>
<div class="tag" id="tag"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ============================ data (injected) ================================
const DESIGNS = __DESIGNS_JSON__;
const RCWA = __RCWA_JSON__;

// =================== physics: JS port of waveguide_physics.py v2 ============
const WL=[450,532,635], VPH=[0.038,0.885,0.217].map((v,_,a)=>v/a.reduce((s,x)=>s+x,0));
const LPROP=20, ACCEPT=0.35, EYEFL=17, RESID=0.10, F0=40;
const deg=d=>d*Math.PI/180, rad2deg=r=>r*180/Math.PI;

function fresnelT(n,fiDeg,pol){
  const ti=deg(fiDeg), ci=Math.cos(ti), st=Math.min(Math.sin(ti)/n,0.9995),
        ct=Math.sqrt(1-st*st);
  const r = pol==="TE" ? (ci-n*ct)/(ci+n*ct) : (n*ci-ct)/(n*ci+ct);
  return 1-r*r;
}
function transmission(d,fiDeg,pol){
  if(pol==="unpol") return .5*(transmission(d,fiDeg,"TE")+transmission(d,fiDeg,"TM"));
  const lam=532, x=Math.sin(deg(fiDeg))+lam/d.period;
  if(x<=1||x>=d.n) return 0;                                  // TIR guiding window
  const ang=Math.asin(Math.min(x/d.n,0.9995));
  const F2=fresnelT(d.n,fiDeg,pol)**2;
  const NB=Math.min(Math.max(LPROP/(2*d.t*Math.tan(ang)),1),60);
  const Tb=Math.exp(-d.alpha*NB*d.t/Math.cos(ang));
  const pb=(4*Math.PI*d.sigma*1e-6*d.n*Math.cos(ang)/(lam*1e-6))**2;
  const Ts=Math.exp(-pb*(1/(1+d.Lc/3e5))*NB);
  const phi=2*Math.PI*d.depth*(d.n-1)/lam;
  const eta=4*(Math.sin(Math.PI*d.duty)/Math.PI)**2*Math.sin(phi/2)**2;
  const acc=Math.exp(-(Math.sin(deg(fiDeg))/ACCEPT)**2);
  return F2*Tb*Ts*(eta*acc)*eta;
}
function anglesRGB(d,fiDeg){        // in-guide angle per wavelength; null if not guided
  return WL.map(w=>{ const x=Math.sin(deg(fiDeg))+w/d.period;
    return (x>1&&x<d.n)?Math.asin(Math.min(x/d.n,.9995)):null; });
}
function fovWindow(d){
  let lo=-1,hi=1;
  WL.forEach(w=>{ lo=Math.max(lo,1-w/d.period); hi=Math.min(hi,d.n-w/d.period); });
  lo=Math.max(-.9995,Math.min(.9995,lo)); hi=Math.max(-.9995,Math.min(.9995,hi));
  return [rad2deg(Math.asin(lo)), rad2deg(Math.asin(hi))];
}
function mtfSystem(d){
  const fc=3/(532e-6*EYEFL), xx=Math.min(F0/fc,.999);
  const mD=(2/Math.PI)*(Math.acos(xx)-xx*Math.sqrt(1-xx*xx));
  const S=1/(1+d.Lc/3e5), blur=(d.sigma/6)**2*S*8e-3;
  const mR=Math.exp(-2*(Math.PI*blur*F0)**2);
  const a=anglesRGB(d,0).map(v=>v??Math.PI/2), dth=a.map(v=>v-a[1]);
  let re=0,im=0; dth.forEach((v,k)=>{ const xk=EYEFL*RESID*v;
    re+=VPH[k]*Math.cos(2*Math.PI*F0*xk); im+=VPH[k]*Math.sin(2*Math.PI*F0*xk); });
  const mC=Math.hypot(re,im);
  const phi=2*Math.PI*d.depth*(d.n-1)/532, mG=1-.15*Math.sin(phi/2)**2;
  const eta=4*(Math.sin(Math.PI*d.duty)/Math.PI)**2*Math.sin(phi/2)**2;
  return mD*mR*mC*mG*(0.80+0.20*eta/0.4053);
}

// ============================== three.js scene ===============================
const W=innerWidth,H=innerHeight;
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setSize(W,H); renderer.setPixelRatio(devicePixelRatio);
document.getElementById("scene").appendChild(renderer.domElement);
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0d1117);
const camera=new THREE.PerspectiveCamera(45,W/H,.1,500);
scene.add(new THREE.AmbientLight(0xffffff,.55));
const key=new THREE.DirectionalLight(0xffffff,.9); key.position.set(12,25,18); scene.add(key);

let az=.9, el=.35, dist=34, target=new THREE.Vector3(10,0,0);
function placeCam(){ camera.position.set(
  target.x+dist*Math.cos(el)*Math.cos(az), target.y+dist*Math.sin(el),
  target.z+dist*Math.cos(el)*Math.sin(az)); camera.lookAt(target); }
placeCam();
let drag=false,px=0,py=0;
addEventListener("mousedown",e=>{ if(e.target.closest(".hud"))return; drag=true;px=e.clientX;py=e.clientY;});
addEventListener("mouseup",()=>drag=false);
addEventListener("mousemove",e=>{ if(!drag)return;
  az+=(e.clientX-px)*.008; el=Math.max(-1.4,Math.min(1.4,el+(e.clientY-py)*.008));
  px=e.clientX;py=e.clientY; placeCam(); });
addEventListener("wheel",e=>{ if(e.target.closest(".hud"))return;
  dist=Math.max(8,Math.min(120,dist*(1+e.deltaY*.001))); placeCam(); },{passive:true});
addEventListener("resize",()=>{ camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix(); renderer.setSize(innerWidth,innerHeight); });

const world=new THREE.Group(); scene.add(world);
const RGBHEX=[0x5aa0ff,0x59d98c,0xff6b5a];
const TSCALE=3;                       // visual thickness exaggeration
const CLEN=3.0;                       // coupler footprint along x (mm)

function ray(points,color,opacity){
  const g=new THREE.BufferGeometry().setFromPoints(points);
  const m=new THREE.LineBasicMaterial({color,transparent:true,opacity});
  return new THREE.Line(g,m);
}

function build(d,fiDeg,pol){
  while(world.children.length) world.remove(world.children[0]);
  const t=d.t*TSCALE, L=LPROP+2*CLEN;

  // PMMA slab
  const slab=new THREE.Mesh(new THREE.BoxGeometry(L,t,7),
    new THREE.MeshPhongMaterial({color:0x9fc5ff,transparent:true,opacity:.16,
      shininess:90,side:THREE.DoubleSide}));
  slab.position.set(L/2-CLEN,-t/2,0); world.add(slab);
  const edges=new THREE.LineSegments(new THREE.EdgesGeometry(slab.geometry),
    new THREE.LineBasicMaterial({color:0x3d4d70}));
  edges.position.copy(slab.position); world.add(edges);

  // gratings: corrugation on the top face (relief exaggerated ×2000)
  const dep=d.depth*1e-6*2000*TSCALE, per=d.period*1e-6, teethPitch=CLEN/24;
  const toothMat=new THREE.MeshPhongMaterial({color:0x5aa9ff});
  [[-CLEN,0],[LPROP,0]].forEach(([x0,_],gi)=>{
    for(let i=0;i<24;i++){
      const tooth=new THREE.Mesh(new THREE.BoxGeometry(teethPitch*d.duty,dep,6.6),toothMat);
      tooth.position.set(x0+i*teethPitch+teethPitch*d.duty/2, dep/2, 0);
      world.add(tooth);
    }
    const lbl=gi===0?"in-coupler":"out-coupler";
  });

  // eye marker
  const eye=new THREE.Mesh(new THREE.SphereGeometry(.9,24,16),
    new THREE.MeshPhongMaterial({color:0xffffff,emissive:0x223355}));
  eye.position.set(LPROP+CLEN/2, 10, 0); world.add(eye);

  // rays: incident -> diffracted TIR zig-zag -> out-coupled
  const angs=anglesRGB(d,fiDeg);
  const si=Math.sin(deg(fiDeg)), inX=-CLEN/2;
  angs.forEach((a,k)=>{
    const c=RGBHEX[k];
    const inc=[new THREE.Vector3(inX-8*si,8,0), new THREE.Vector3(inX,0,0)];
    world.add(ray(inc,c,.95));
    if(a===null){                                     // NOT GUIDED, leak fan
      world.add(ray([new THREE.Vector3(inX,0,0),
        new THREE.Vector3(inX+4,-t-3,0)],c,.25));
      world.add(ray([new THREE.Vector3(inX,0,0),
        new THREE.Vector3(inX+6,-t-1.5,0)],c,.15));
      return;
    }
    const dx=t*Math.tan(a);                           // half-bounce advance
    const pts=[new THREE.Vector3(inX,0,0)];
    let x=inX,y=0,down=true,guard=0;
    while(x<LPROP+CLEN/2&&guard++<300){
      x+=dx; y=down?-t:0; pts.push(new THREE.Vector3(x,y,0)); down=!down;
      if(x>=LPROP&&y===0) break;                      // reached out-coupler on top face
    }
    const T=transmission(d,fiDeg,pol==="unpol"?"TE":pol);
    pts.push(new THREE.Vector3(x+8*si,10,0));         // out-coupled to eye
    world.add(ray(pts,c,Math.max(.25,Math.min(.95,T*6))));
  });

  // axes hint
  const ax=new THREE.AxesHelper(2.2); ax.position.set(-CLEN-3,-t-2,-4); world.add(ax);
}

// ============================== HUD wiring ===================================
const sel=document.getElementById("design");
DESIGNS.forEach((d,i)=>{ const o=document.createElement("option"); o.value=i;
  o.textContent=`#${d.rank}  Λ=${d.period.toFixed(0)}nm d=${d.depth.toFixed(0)}nm `+
    `duty=${d.duty.toFixed(2)} t=${d.t.toFixed(2)}mm`; sel.appendChild(o); });
let pol="unpol";
document.getElementById("polseg").addEventListener("click",e=>{
  if(e.target.dataset.p){ pol=e.target.dataset.p;
    document.querySelectorAll("#polseg button").forEach(b=>b.classList.toggle("on",b===e.target));
    refresh(); }});
document.getElementById("fov").addEventListener("input",refresh);
sel.addEventListener("change",refresh);

function row(k,v,cls){ return `<tr><td>${k}</td><td class="${cls||''}">${v}</td></tr>`; }

function refresh(){
  const d=DESIGNS[+sel.value], fi=+document.getElementById("fov").value;
  document.getElementById("v_fov").textContent=fi.toFixed(1)+"°";
  build(d,fi,pol);

  const T0=transmission(d,0,pol), Tf=transmission(d,fi,pol);
  const Tte=transmission(d,fi,"TE"), Ttm=transmission(d,fi,"TM");
  const [flo,fhi]=fovWindow(d);
  const a=anglesRGB(d,0), chrom=(a[2]!==null&&a[0]!==null)?rad2deg(a[2]-a[0]):NaN;
  const guided=WL.map((w,k)=>a[k]!==null);
  document.getElementById("perf").innerHTML=
    row("System MTF @40 cyc/mm", mtfSystem(d).toFixed(4))+
    row(`T (${pol}) @ θᵢ=0°`, (100*T0).toFixed(2)+"%")+
    row(`T (${pol}) @ θᵢ=${fi.toFixed(1)}°`, (100*Tf).toFixed(2)+"%", Tf===0?"bad":"")+
    row("T_TE / T_TM @ θᵢ", (100*Tte).toFixed(2)+"% / "+(100*Ttm).toFixed(2)+"%")+
    row("diattenuation (TE−TM)/(TE+TM)", ((Tte-Ttm)/(Tte+Ttm+1e-12)).toFixed(3))+
    row("chromatic spread (in-guide)", isNaN(chrom)?", ":chrom.toFixed(1)+"°")+
    row("full-RGB guided FOV window", `[${flo.toFixed(1)}°, ${fhi.toFixed(1)}°]`,
        (fhi-flo)>0?"good":"bad")+
    row("RGB guided @ θᵢ=0", guided.map((g,k)=>g?"✓":"✗").join(" "),
        guided.every(Boolean)?"good":"bad")+
    row("TIR bounces (green)", a[1]===null?", ":
        Math.min(Math.max(LPROP/(2*d.t*Math.tan(a[1])),1),60).toFixed(1));

  const rc=RCWA[String(d.rank)]||[];
  document.getElementById("rcwa").innerHTML = rc.length ?
    rc.map(r=>row(`η₁ @ ${r.lam.toFixed(0)}nm  scalar ${ (100*r.scalar).toFixed(1)}%`,
      `TE ${(100*r.TE).toFixed(1)}% / TM ${(100*r.TM).toFixed(1)}%`)).join("")
    : row("no RCWA data","run rigorous_solver.py --designs","warn");
  document.getElementById("rcwacap").textContent = rc.length ?
    "Exact vector solutions of Maxwell's equations (grcwa RCWA). TE≠TM shows "+
    "true electromagnetic polarization splitting that scalar theory cannot capture." : "";

  drawAngPlot(d);
  document.getElementById("tag").innerHTML=
    `design #${d.rank} · n=${d.n.toFixed(3)} α=${d.alpha.toExponential(1)}/mm `+
    `σ=${d.sigma.toFixed(2)}nm L_c=${(d.Lc/1e3).toFixed(0)}µm · physics v2`;
}

function drawAngPlot(d){
  const cv=document.getElementById("angplot"),g=cv.getContext("2d");
  g.clearRect(0,0,cv.width,cv.height);
  const X=a=>((a+20)/40)*(cv.width-34)+28, Y=v=>cv.height-16-v*(cv.height-30);
  let peak=1e-9;
  for(let a=-20;a<=20;a+=.25) peak=Math.max(peak,transmission(d,a,"TE"));
  const [flo,fhi]=fovWindow(d);
  if(fhi>flo){ g.fillStyle="rgba(89,217,140,.12)";
    g.fillRect(X(Math.max(flo,-20)),8,X(Math.min(fhi,20))-X(Math.max(flo,-20)),cv.height-24); }
  g.strokeStyle="#33405a"; g.strokeRect(28,8,cv.width-34,cv.height-24);
  g.fillStyle="#8b96ab"; g.font="9px sans-serif";
  [-20,-10,0,10,20].forEach(a=>{ g.fillText(a+"°",X(a)-6,cv.height-4); });
  g.fillText((100*peak).toFixed(1)+"%",2,14); g.fillText("0",16,cv.height-16);
  const plots=[["TE",[],"#5aa9ff",false],["TM",[],"#ffb84d",true],["unpol",[],"#e8ecf5",false]];
  plots.forEach(([p,pts,col,dash],pi)=>{
    g.strokeStyle=col; g.lineWidth=p==="unpol"?2:1.1;
    g.setLineDash(dash?[4,3]:[]); g.beginPath(); let started=false;
    for(let a=-20;a<=20;a+=.25){
      const v=transmission(d,a,p)/peak, x=X(a), y=Y(Math.min(v,1));
      started?g.lineTo(x,y):(g.moveTo(x,y),started=true);
    }
    g.stroke(); g.setLineDash([]);
  });
}

refresh();
(function loop(){ requestAnimationFrame(loop); renderer.render(scene,camera); })();
</script>
</body>
</html>
"""


def main():
    designs = load_designs()
    rcwa = load_rcwa()
    html = (TEMPLATE
            .replace("__DESIGNS_JSON__", json.dumps(designs))
            .replace("__RCWA_JSON__", json.dumps(rcwa)))
    with open(OUT_HTML, "w") as f:
        f.write(html)
    print(f"written -> {OUT_HTML}  ({len(designs)} designs, "
          f"RCWA data for {len(rcwa)} ranks)")


if __name__ == "__main__":
    main()
